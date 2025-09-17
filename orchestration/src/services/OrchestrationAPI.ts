/**
 * Main Orchestration API Gateway
 *
 * Type-safe REST API that coordinates between TypeScript orchestration layer
 * and Python uncertainty quantification services. Handles request routing,
 * load balancing, auto-scaling decisions, and real-time monitoring.
 */

import express, { Express, Request, Response, NextFunction } from 'express';
import { Server as SocketIOServer } from 'socket.io';
import { createServer } from 'http';
import Joi from 'joi';
import { v4 as uuidv4 } from 'uuid';
import { Logger } from 'winston';
import axios from 'axios';

import {
  OrchestrationConfig,
  UncertaintyRequest,
  UncertaintyResponse,
  CalibrationRequest,
  CalibrationResponse,
  HealthCheckResponse,
  DashboardMetrics,
  NodeMetrics,
  WebSocketEvent,
} from '../types/calibration';

import { LoadBalancer } from './LoadBalancer';
import { AutoScaler } from './AutoScaler';
import { MessageBroker } from './MessageBroker';
import { createLogger } from '../utils/logger';
import { CircuitBreaker } from '../utils/CircuitBreaker';

export class OrchestrationAPI {
  private app: Express;
  private server;
  private io: SocketIOServer;
  private logger: Logger;
  private loadBalancer: LoadBalancer;
  private autoScaler: AutoScaler;
  private messageBroker: MessageBroker;
  private circuitBreaker: CircuitBreaker;

  // Request tracking
  private activeRequests: Map<string, { startTime: number; type: string }> = new Map();
  private requestMetrics = {
    totalRequests: 0,
    totalErrors: 0,
    avgResponseTime: 0,
    requestsPerSecond: 0,
    lastRequestTime: Date.now(),
  };

  constructor(private config: OrchestrationConfig) {
    this.logger = createLogger('OrchestrationAPI');
    this.app = express();
    this.server = createServer(this.app);
    this.io = new SocketIOServer(this.server, {
      cors: {
        origin: "*",
        methods: ["GET", "POST"]
      }
    });

    // Initialize services
    this.loadBalancer = new LoadBalancer(config.loadBalancer);
    this.autoScaler = new AutoScaler(config.autoScaler);
    this.messageBroker = new MessageBroker({
      redisUrl: config.redisUrl,
      retryAttempts: 3,
      retryDelayMs: 1000,
      deadLetterQueueEnabled: true,
      eventTtlSeconds: 3600,
    });
    this.circuitBreaker = new CircuitBreaker(config.circuitBreaker);

    this.setupMiddleware();
    this.setupRoutes();
    this.setupWebSocket();
    this.setupEventHandlers();
  }

  /**
   * Start the orchestration API server
   */
  public async start(): Promise<void> {
    try {
      await this.messageBroker.initialize();
      this.setupPythonServiceDiscovery();

      this.server.listen(this.config.port, () => {
        this.logger.info(`Orchestration API started on port ${this.config.port}`, {
          clusterId: this.config.clusterId,
          region: this.config.region,
        });
      });

      // Start background tasks
      this.startMetricsCollection();
      this.startAutoScalingEvaluation();

    } catch (error) {
      this.logger.error('Failed to start Orchestration API', { error });
      throw error;
    }
  }

  /**
   * Graceful shutdown
   */
  public async shutdown(): Promise<void> {
    this.logger.info('Shutting down Orchestration API...');

    this.server.close();
    await this.messageBroker.shutdown();
    this.loadBalancer.shutdown();
    this.autoScaler.shutdown();

    this.logger.info('Orchestration API shutdown complete');
  }

  // Private methods

  private setupMiddleware(): void {
    this.app.use(express.json({ limit: '10mb' }));
    this.app.use(express.urlencoded({ extended: true }));

    // Request logging
    this.app.use((req: Request, res: Response, next: NextFunction) => {
      const requestId = uuidv4();
      req.headers['x-request-id'] = requestId;

      this.logger.debug('Incoming request', {
        requestId,
        method: req.method,
        path: req.path,
        userAgent: req.get('User-Agent'),
      });

      next();
    });

    // Error handling
    this.app.use((error: Error, req: Request, res: Response, next: NextFunction) => {
      this.logger.error('Unhandled error', {
        requestId: req.headers['x-request-id'],
        error: error.message,
        stack: error.stack,
      });

      res.status(500).json({
        error: 'Internal server error',
        requestId: req.headers['x-request-id'],
      });
    });
  }

  private setupRoutes(): void {
    // Health check
    this.app.get('/health', async (req: Request, res: Response) => {
      const health = await this.getHealthStatus();
      const statusCode = health.status === 'healthy' ? 200 : 503;
      res.status(statusCode).json(health);
    });

    // Uncertainty generation endpoint
    this.app.post('/generate', async (req: Request, res: Response) => {
      const requestId = req.headers['x-request-id'] as string;

      try {
        const validatedRequest = await this.validateUncertaintyRequest(req.body, requestId);
        const result = await this.processUncertaintyRequest(validatedRequest);

        res.json(result);
      } catch (error) {
        this.handleRequestError(error as Error, requestId, res);
      }
    });

    // Batch uncertainty generation
    this.app.post('/generate/batch', async (req: Request, res: Response) => {
      const requestId = req.headers['x-request-id'] as string;

      try {
        const requests = req.body.requests as UncertaintyRequest[];
        const results = await this.processBatchUncertaintyRequests(requests, requestId);

        res.json({ results, requestId });
      } catch (error) {
        this.handleRequestError(error as Error, requestId, res);
      }
    });

    // Calibration validation
    this.app.post('/calibrate', async (req: Request, res: Response) => {
      const requestId = req.headers['x-request-id'] as string;

      try {
        const validatedRequest = await this.validateCalibrationRequest(req.body, requestId);
        const result = await this.processCalibrationRequest(validatedRequest);

        res.json(result);
      } catch (error) {
        this.handleRequestError(error as Error, requestId, res);
      }
    });

    // Cluster metrics and status
    this.app.get('/cluster/status', (req: Request, res: Response) => {
      const status = this.getClusterStatus();
      res.json(status);
    });

    // Scaling recommendations
    this.app.get('/cluster/scaling', (req: Request, res: Response) => {
      const recommendations = this.getScalingRecommendations();
      res.json(recommendations);
    });

    // Load balancer statistics
    this.app.get('/cluster/load-balancer', (req: Request, res: Response) => {
      const stats = this.loadBalancer.getStatistics();
      res.json(stats);
    });

    // Auto-scaler statistics
    this.app.get('/cluster/auto-scaler', (req: Request, res: Response) => {
      const stats = this.autoScaler.getScalingStatistics();
      res.json(stats);
    });

    // Message broker metrics
    this.app.get('/cluster/message-broker', (req: Request, res: Response) => {
      const metrics = this.messageBroker.getMetrics();
      res.json(metrics);
    });

    // Dashboard data endpoint
    this.app.get('/dashboard/metrics', (req: Request, res: Response) => {
      const metrics = this.getDashboardMetrics();
      res.json(metrics);
    });
  }

  private setupWebSocket(): void {
    this.io.on('connection', (socket) => {
      this.logger.info('WebSocket client connected', { socketId: socket.id });

      socket.emit('connected', {
        clusterId: this.config.clusterId,
        region: this.config.region,
        timestamp: Date.now(),
      });

      // Subscribe to real-time updates
      socket.on('subscribe', (topics: string[]) => {
        topics.forEach(topic => {
          socket.join(topic);
          this.logger.debug('Client subscribed to topic', { socketId: socket.id, topic });
        });
      });

      socket.on('disconnect', () => {
        this.logger.info('WebSocket client disconnected', { socketId: socket.id });
      });
    });
  }

  private setupEventHandlers(): void {
    // Load balancer events
    this.loadBalancer.on('nodeHealthChanged', (event) => {
      this.broadcastWebSocketEvent({
        type: 'NODE_STATUS_CHANGE',
        timestamp: Date.now(),
        data: event,
        severity: event.isHealthy ? 'info' : 'warning',
      });
    });

    // Auto-scaler events
    this.autoScaler.on('scalingAction', (event) => {
      this.broadcastWebSocketEvent({
        type: 'SCALING_EVENT',
        timestamp: Date.now(),
        data: event,
        severity: 'info',
      });
    });

    // Message broker events
    this.messageBroker.on('eventPublished', (event) => {
      this.logger.debug('Event published via message broker', event);
    });
  }

  private async validateUncertaintyRequest(body: any, requestId: string): Promise<UncertaintyRequest> {
    const schema = Joi.object({
      text: Joi.string().required().max(10000),
      maxLength: Joi.number().integer().min(1).max(500).default(50),
      numReturnSequences: Joi.number().integer().min(1).max(10).default(1),
      temperature: Joi.number().min(0.1).max(2.0).default(1.0),
      pbaConfig: Joi.object().optional(),
      clientId: Joi.string().optional(),
    });

    const { error, value } = schema.validate(body);
    if (error) {
      throw new Error(`Validation error: ${error.details[0].message}`);
    }

    return {
      ...value,
      requestId,
    };
  }

  private async validateCalibrationRequest(body: any, requestId: string): Promise<CalibrationRequest> {
    const schema = Joi.object({
      uncertaintyScores: Joi.array().items(Joi.number().min(0).max(1)).required(),
      correctnessLabels: Joi.array().items(Joi.number().integer().min(0).max(1)).required(),
      datasetName: Joi.string().optional(),
    });

    const { error, value } = schema.validate(body);
    if (error) {
      throw new Error(`Validation error: ${error.details[0].message}`);
    }

    // Validate array lengths match
    if (value.uncertaintyScores.length !== value.correctnessLabels.length) {
      throw new Error('uncertaintyScores and correctnessLabels arrays must have the same length');
    }

    return {
      ...value,
      requestId,
    };
  }

  private async processUncertaintyRequest(request: UncertaintyRequest): Promise<UncertaintyResponse> {
    const startTime = Date.now();
    this.activeRequests.set(request.requestId, { startTime, type: 'uncertainty' });

    try {
      // Select optimal node using load balancer
      const selectedNode = await this.loadBalancer.selectNode(request);
      if (!selectedNode) {
        throw new Error('No healthy nodes available');
      }

      // Make request to Python service through circuit breaker
      const response = await this.circuitBreaker.execute(async () => {
        const pythonResponse = await axios.post(
          `${selectedNode.endpoint}/generate`,
          request,
          { timeout: this.config.loadBalancer.timeoutMs }
        );

        return pythonResponse.data;
      });

      // Record success
      this.loadBalancer.recordSuccess(selectedNode.nodeId);
      this.updateRequestMetrics(Date.now() - startTime, false);

      // Publish event about successful request
      await this.messageBroker.publishEvent('orchestration_events', {
        eventType: 'UPDATE_CALIBRATION',
        nodeId: selectedNode.nodeId,
        data: {
          requestId: request.requestId,
          processingTime: Date.now() - startTime,
          uncertainty: response.uncertaintyScores?.[0] || 0,
        },
      });

      return response;

    } catch (error) {
      this.logger.error('Failed to process uncertainty request', {
        requestId: request.requestId,
        error: (error as Error).message,
      });

      this.updateRequestMetrics(Date.now() - startTime, true);
      throw error;

    } finally {
      this.activeRequests.delete(request.requestId);
    }
  }

  private async processBatchUncertaintyRequests(
    requests: UncertaintyRequest[],
    batchRequestId: string
  ): Promise<UncertaintyResponse[]> {
    // Add request IDs if not provided
    const requestsWithIds = requests.map(req => ({
      ...req,
      requestId: req.requestId || uuidv4(),
    }));

    // Process requests concurrently with load balancing
    const results = await Promise.allSettled(
      requestsWithIds.map(req => this.processUncertaintyRequest(req))
    );

    // Convert results and handle failures
    return results.map((result, index) => {
      if (result.status === 'fulfilled') {
        return result.value;
      } else {
        this.logger.error('Batch request failed', {
          batchRequestId,
          requestIndex: index,
          error: result.reason?.message,
        });

        // Return error response
        return {
          requestId: requestsWithIds[index].requestId,
          generatedTexts: [],
          uncertaintyScores: [],
          tokenUncertainties: [],
          metadata: {
            modelName: 'unknown',
            nodeId: 'unknown',
            partitionId: 'unknown',
            calibrationParameters: {},
          },
          processingTimeMs: 0,
          error: result.reason?.message || 'Unknown error',
        } as UncertaintyResponse & { error: string };
      }
    });
  }

  private async processCalibrationRequest(request: CalibrationRequest): Promise<CalibrationResponse> {
    const startTime = Date.now();

    try {
      // Select node for calibration processing
      const selectedNode = await this.loadBalancer.selectNode({
        text: 'calibration',
        requestId: request.requestId,
        maxLength: 1,
        numReturnSequences: 1,
        temperature: 1.0,
      });

      if (!selectedNode) {
        throw new Error('No healthy nodes available for calibration');
      }

      const response = await axios.post(
        `${selectedNode.endpoint}/calibrate`,
        request,
        { timeout: this.config.loadBalancer.timeoutMs }
      );

      this.loadBalancer.recordSuccess(selectedNode.nodeId);
      return response.data;

    } catch (error) {
      this.logger.error('Failed to process calibration request', {
        requestId: request.requestId,
        error: (error as Error).message,
      });
      throw error;
    }
  }

  private setupPythonServiceDiscovery(): void {
    // In production, this would integrate with service discovery (Consul, Kubernetes, etc.)
    // For now, add configured Python service endpoint
    this.loadBalancer.addNode(
      'python-service-1',
      this.config.pythonServiceUrl,
      this.config.region
    );

    this.logger.info('Python service registered with load balancer', {
      endpoint: this.config.pythonServiceUrl,
    });
  }

  private async getHealthStatus(): Promise<HealthCheckResponse> {
    const messageBrokerHealth = await this.messageBroker.healthCheck();
    const loadBalancerStats = this.loadBalancer.getStatistics();

    const checks = {
      messageBroker: messageBrokerHealth.status === 'healthy',
      loadBalancer: loadBalancerStats.healthyNodes > 0,
      circuitBreaker: this.circuitBreaker.getState() !== 'OPEN',
      activeRequests: this.activeRequests.size < 1000, // Arbitrary threshold
    };

    const allHealthy = Object.values(checks).every(Boolean);

    return {
      status: allHealthy ? 'healthy' : 'unhealthy',
      timestamp: Date.now(),
      nodeId: `orchestrator-${this.config.region}`,
      checks,
      metadata: {
        activeRequests: this.activeRequests.size,
        totalRequests: this.requestMetrics.totalRequests,
        errorRate: this.requestMetrics.totalErrors / Math.max(1, this.requestMetrics.totalRequests),
      },
    };
  }

  private getClusterStatus(): any {
    const loadBalancerStats = this.loadBalancer.getStatistics();
    const autoScalerStats = this.autoScaler.getScalingStatistics();
    const messageBrokerMetrics = this.messageBroker.getMetrics();

    return {
      clusterId: this.config.clusterId,
      region: this.config.region,
      timestamp: Date.now(),
      nodes: {
        total: loadBalancerStats.totalNodes,
        healthy: loadBalancerStats.healthyNodes,
      },
      requests: this.requestMetrics,
      scaling: {
        lastAction: autoScalerStats.lastAction,
        totalActions: autoScalerStats.totalActions,
      },
      messaging: {
        totalPublished: messageBrokerMetrics.totalPublished,
        totalReceived: messageBrokerMetrics.totalReceived,
        errors: messageBrokerMetrics.totalErrors,
      },
    };
  }

  private getScalingRecommendations(): any {
    // This would get real cluster state and node metrics in production
    const mockClusterState = {
      clusterId: this.config.clusterId,
      totalNodes: 3,
      activeNodes: 3,
      partitions: {},
      leaders: {},
      loadDistribution: {},
      scalingDecision: null,
      lastScalingAction: Date.now(),
    };

    const mockNodeMetrics = {}; // Would be populated from real metrics

    return this.autoScaler.getScalingRecommendations(mockClusterState, mockNodeMetrics);
  }

  private getDashboardMetrics(): DashboardMetrics {
    const loadBalancerStats = this.loadBalancer.getStatistics();

    return {
      timestamp: Date.now(),
      clusterId: this.config.clusterId,
      totalNodes: loadBalancerStats.totalNodes,
      activeNodes: loadBalancerStats.healthyNodes,
      totalRequests: this.requestMetrics.totalRequests,
      totalErrors: this.requestMetrics.totalErrors,
      avgLatency: this.requestMetrics.avgResponseTime,
      avgUncertainty: 0.5, // Would calculate from real data
      avgCalibrationQuality: 0.95, // Would calculate from real data
      scalingDecision: null,
      nodeMetrics: [], // Would populate from real node metrics
    };
  }

  private updateRequestMetrics(responseTime: number, isError: boolean): void {
    this.requestMetrics.totalRequests++;
    if (isError) {
      this.requestMetrics.totalErrors++;
    }

    // Update rolling average response time
    const alpha = 0.1;
    this.requestMetrics.avgResponseTime =
      this.requestMetrics.avgResponseTime * (1 - alpha) + responseTime * alpha;

    this.requestMetrics.lastRequestTime = Date.now();
  }

  private broadcastWebSocketEvent(event: WebSocketEvent): void {
    this.io.emit('event', event);
  }

  private startMetricsCollection(): void {
    setInterval(() => {
      const metrics = this.getDashboardMetrics();
      this.broadcastWebSocketEvent({
        type: 'METRICS_UPDATE',
        timestamp: Date.now(),
        data: metrics,
      });
    }, 5000); // Every 5 seconds
  }

  private startAutoScalingEvaluation(): void {
    this.autoScaler.on('evaluationCycle', () => {
      // In production, would get real cluster state and trigger scaling decisions
      this.logger.debug('Auto-scaling evaluation cycle triggered');
    });
  }

  private handleRequestError(error: Error, requestId: string, res: Response): void {
    this.logger.error('Request processing error', { requestId, error: error.message });

    let statusCode = 500;
    let message = 'Internal server error';

    if (error.message.includes('Validation error')) {
      statusCode = 400;
      message = error.message;
    } else if (error.message.includes('No healthy nodes')) {
      statusCode = 503;
      message = 'Service temporarily unavailable';
    }

    res.status(statusCode).json({
      error: message,
      requestId,
      timestamp: Date.now(),
    });
  }
}