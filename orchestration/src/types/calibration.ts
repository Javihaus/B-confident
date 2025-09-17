/**
 * Core Types for Distributed Calibration System
 *
 * Strong typing for all distributed calibration events, state management,
 * and inter-service communication contracts.
 */

export type NodeRole = 'follower' | 'candidate' | 'leader' | 'observer';

export type PartitionStrategy = 'hash_based' | 'geographic' | 'load_based' | 'consistency_hash';

export type ScalingDecision = 'scale_up' | 'scale_down' | 'rebalance' | 'no_action';

export type CalibrationEventType =
  | 'UPDATE_CALIBRATION'
  | 'REQUEST_SYNC'
  | 'NODE_STATUS'
  | 'THRESHOLD_CHANGE'
  | 'RECALIBRATION_TRIGGER'
  | 'PARTITION_EVENT'
  | 'LEADER_ELECTION'
  | 'LOAD_BALANCE'
  | 'AUTO_SCALE'
  | 'HEALTH_CHECK'
  | 'FAILOVER';

export interface NodeMetrics {
  nodeId: string;
  role: NodeRole;
  load: number; // 0.0 - 1.0
  throughputRps: number;
  latencyMs: number;
  memoryUsage: number; // 0.0 - 1.0
  cpuUsage: number; // 0.0 - 1.0
  errorRate: number; // 0.0 - 1.0
  lastHeartbeat: number;
  partitionId: string;
  region: string;
}

export interface CalibrationState {
  nodeId: string;
  uncertaintyHistory: number[];
  accuracyHistory: number[];
  calibrationParameters: Record<string, number>;
  lastUpdated: number;
  calibrationQuality: number;
  sampleCount: number;
}

export interface PartitionInfo {
  partitionId: string;
  strategy: PartitionStrategy;
  nodeIds: string[];
  leaderId: string | null;
  dataRange: [unknown, unknown];
  replicaCount: number;
  lastRebalance: number;
}

export interface ClusterState {
  clusterId: string;
  totalNodes: number;
  activeNodes: number;
  partitions: Record<string, PartitionInfo>;
  leaders: Record<string, string>; // partitionId -> leaderId
  loadDistribution: Record<string, number>; // nodeId -> load
  scalingDecision: ScalingDecision | null;
  lastScalingAction: number;
}

export interface CalibrationEvent {
  eventType: CalibrationEventType;
  nodeId: string;
  timestamp: number;
  data: Record<string, unknown>;
  sequenceId: number;
}

export interface GlobalCalibrationState {
  globalParameters: Record<string, number>;
  nodeStates: Record<string, CalibrationState>;
  nodeMetrics: Record<string, NodeMetrics>;
  clusterState: ClusterState;
  lastGlobalUpdate: number;
  consensusThreshold: number;
  updateFrequency: number;
  partitionStrategy: PartitionStrategy;
}

// Request/Response types for Python-TS communication
export interface UncertaintyRequest {
  text: string;
  maxLength: number;
  numReturnSequences: number;
  temperature: number;
  pbaConfig?: Record<string, unknown>;
  requestId: string;
  clientId?: string;
}

export interface UncertaintyResponse {
  requestId: string;
  generatedTexts: string[];
  uncertaintyScores: number[];
  tokenUncertainties: number[][];
  metadata: ResponseMetadata;
  processingTimeMs: number;
}

export interface ResponseMetadata {
  modelName: string;
  nodeId: string;
  partitionId: string;
  calibrationParameters: Record<string, number>;
  debugInfo?: Record<string, unknown>;
}

export interface CalibrationRequest {
  uncertaintyScores: number[];
  correctnessLabels: number[];
  datasetName?: string;
  requestId: string;
}

export interface CalibrationResponse {
  requestId: string;
  ece: number;
  brierScore: number;
  auroc: number;
  stabilityScore: number;
  complianceStatus: string;
  nodeId: string;
}

// Auto-scaling configuration
export interface AutoScalerConfig {
  minNodes: number;
  maxNodes: number;
  scaleUpThreshold: number;
  scaleDownThreshold: number;
  cooldownPeriod: number;
  evaluationInterval: number;
}

// Load balancer configuration
export interface LoadBalancerConfig {
  strategy: 'round_robin' | 'weighted' | 'least_loaded' | 'geographic';
  healthCheckInterval: number;
  maxRetries: number;
  timeoutMs: number;
}

// Circuit breaker configuration
export interface CircuitBreakerConfig {
  failureThreshold: number;
  timeout: number;
  halfOpenTimeout: number;
  monitoringWindow: number;
}

// Orchestration configuration
export interface OrchestrationConfig {
  port: number;
  redisUrl: string;
  pythonServiceUrl: string;
  clusterId: string;
  region: string;
  autoScaler: AutoScalerConfig;
  loadBalancer: LoadBalancerConfig;
  circuitBreaker: CircuitBreakerConfig;
  enableMetrics: boolean;
  enableDashboard: boolean;
  logLevel: 'error' | 'warn' | 'info' | 'debug';
}

// Health check response
export interface HealthCheckResponse {
  status: 'healthy' | 'unhealthy' | 'degraded';
  timestamp: number;
  nodeId: string;
  checks: Record<string, boolean>;
  metadata?: Record<string, unknown>;
}

// Scaling recommendation
export interface ScalingRecommendation {
  decision: ScalingDecision;
  reason: string;
  confidence: number; // 0.0 - 1.0
  recommendedActions: Array<{
    action: string;
    priority: 'low' | 'medium' | 'high';
    description: string;
  }>;
  clusterLoadStats: {
    avgLoad: number;
    maxLoad: number;
    minLoad: number;
    loadVariance: number;
  };
}

// Dashboard metrics
export interface DashboardMetrics {
  timestamp: number;
  clusterId: string;
  totalNodes: number;
  activeNodes: number;
  totalRequests: number;
  totalErrors: number;
  avgLatency: number;
  avgUncertainty: number;
  avgCalibrationQuality: number;
  scalingDecision: ScalingDecision | null;
  nodeMetrics: NodeMetrics[];
}

// WebSocket event types for real-time dashboard
export type WebSocketEventType =
  | 'METRICS_UPDATE'
  | 'SCALING_EVENT'
  | 'NODE_STATUS_CHANGE'
  | 'CALIBRATION_UPDATE'
  | 'ERROR_ALERT'
  | 'PERFORMANCE_ALERT';

export interface WebSocketEvent {
  type: WebSocketEventType;
  timestamp: number;
  data: unknown;
  nodeId?: string;
  severity?: 'info' | 'warning' | 'error';
}