/**
 * TypeScript Load Balancer for Distributed Uncertainty Quantification
 *
 * Provides intelligent load balancing with strong typing, health monitoring,
 * and multiple balancing strategies optimized for uncertainty workloads.
 */

import { EventEmitter } from 'events';
import { Logger } from 'winston';
import { NodeMetrics, LoadBalancerConfig, UncertaintyRequest } from '../types/calibration';
import { createLogger } from '../utils/logger';

export interface LoadBalancerNode {
  nodeId: string;
  endpoint: string;
  region: string;
  isHealthy: boolean;
  lastHealthCheck: number;
  consecutiveFailures: number;
  weight: number;
}

export class LoadBalancer extends EventEmitter {
  private nodes: Map<string, LoadBalancerNode> = new Map();
  private nodeMetrics: Map<string, NodeMetrics> = new Map();
  private roundRobinIndex = 0;
  private logger: Logger;
  private healthCheckInterval?: NodeJS.Timeout;

  constructor(private config: LoadBalancerConfig) {
    super();
    this.logger = createLogger('LoadBalancer');
    this.startHealthChecking();
  }

  /**
   * Add a node to the load balancer
   */
  public addNode(nodeId: string, endpoint: string, region: string): void {
    const node: LoadBalancerNode = {
      nodeId,
      endpoint,
      region,
      isHealthy: true,
      lastHealthCheck: Date.now(),
      consecutiveFailures: 0,
      weight: 1.0,
    };

    this.nodes.set(nodeId, node);
    this.logger.info(`Added node to load balancer`, { nodeId, endpoint, region });
    this.emit('nodeAdded', node);
  }

  /**
   * Remove a node from the load balancer
   */
  public removeNode(nodeId: string): boolean {
    const removed = this.nodes.delete(nodeId);
    this.nodeMetrics.delete(nodeId);

    if (removed) {
      this.logger.info(`Removed node from load balancer`, { nodeId });
      this.emit('nodeRemoved', nodeId);
    }

    return removed;
  }

  /**
   * Update node metrics for intelligent load balancing
   */
  public updateNodeMetrics(nodeId: string, metrics: NodeMetrics): void {
    this.nodeMetrics.set(nodeId, metrics);

    const node = this.nodes.get(nodeId);
    if (node) {
      // Calculate weight based on node performance
      const loadWeight = Math.max(0.1, 1.0 - metrics.load);
      const latencyWeight = Math.max(0.1, 1.0 - Math.min(1.0, metrics.latencyMs / 1000));
      const errorWeight = Math.max(0.1, 1.0 - metrics.errorRate);

      // Combined weight for load balancing
      node.weight = (loadWeight * 0.5 + latencyWeight * 0.3 + errorWeight * 0.2);

      // Update health status based on metrics
      const isHealthy = this.isNodeHealthy(metrics);
      if (node.isHealthy !== isHealthy) {
        node.isHealthy = isHealthy;
        this.emit('nodeHealthChanged', { nodeId, isHealthy, metrics });
      }
    }
  }

  /**
   * Select optimal node for handling a request
   */
  public async selectNode(
    request: UncertaintyRequest,
    excludeNodes: string[] = []
  ): Promise<LoadBalancerNode | null> {
    const availableNodes = Array.from(this.nodes.values()).filter(
      (node) =>
        node.isHealthy &&
        !excludeNodes.includes(node.nodeId)
    );

    if (availableNodes.length === 0) {
      this.logger.warn('No healthy nodes available for load balancing');
      return null;
    }

    let selectedNode: LoadBalancerNode;

    switch (this.config.strategy) {
      case 'weighted':
        selectedNode = this.selectWeightedNode(availableNodes);
        break;
      case 'least_loaded':
        selectedNode = this.selectLeastLoadedNode(availableNodes);
        break;
      case 'geographic':
        selectedNode = this.selectGeographicNode(availableNodes, request);
        break;
      default: // round_robin
        selectedNode = this.selectRoundRobinNode(availableNodes);
        break;
    }

    this.logger.debug('Selected node for request', {
      nodeId: selectedNode.nodeId,
      strategy: this.config.strategy,
      requestId: request.requestId,
    });

    return selectedNode;
  }

  /**
   * Get current load balancer statistics
   */
  public getStatistics(): {
    totalNodes: number;
    healthyNodes: number;
    strategy: string;
    roundRobinIndex: number;
    nodeStatistics: Array<{
      nodeId: string;
      isHealthy: boolean;
      weight: number;
      consecutiveFailures: number;
      metrics?: NodeMetrics;
    }>;
  } {
    const nodeStats = Array.from(this.nodes.values()).map((node) => ({
      nodeId: node.nodeId,
      isHealthy: node.isHealthy,
      weight: node.weight,
      consecutiveFailures: node.consecutiveFailures,
      metrics: this.nodeMetrics.get(node.nodeId),
    }));

    return {
      totalNodes: this.nodes.size,
      healthyNodes: nodeStats.filter((n) => n.isHealthy).length,
      strategy: this.config.strategy,
      roundRobinIndex: this.roundRobinIndex,
      nodeStatistics: nodeStats,
    };
  }

  /**
   * Record a successful request to a node
   */
  public recordSuccess(nodeId: string): void {
    const node = this.nodes.get(nodeId);
    if (node) {
      node.consecutiveFailures = 0;
      if (!node.isHealthy) {
        node.isHealthy = true;
        this.logger.info(`Node recovered and marked as healthy`, { nodeId });
        this.emit('nodeHealthChanged', { nodeId, isHealthy: true });
      }
    }
  }

  /**
   * Record a failed request to a node
   */
  public recordFailure(nodeId: string): void {
    const node = this.nodes.get(nodeId);
    if (node) {
      node.consecutiveFailures++;

      if (node.consecutiveFailures >= this.config.maxRetries && node.isHealthy) {
        node.isHealthy = false;
        this.logger.warn(`Node marked as unhealthy after ${node.consecutiveFailures} failures`, { nodeId });
        this.emit('nodeHealthChanged', { nodeId, isHealthy: false });
      }
    }
  }

  /**
   * Shutdown the load balancer
   */
  public shutdown(): void {
    if (this.healthCheckInterval) {
      clearInterval(this.healthCheckInterval);
    }
    this.removeAllListeners();
    this.logger.info('Load balancer shutdown complete');
  }

  // Private methods

  private selectRoundRobinNode(nodes: LoadBalancerNode[]): LoadBalancerNode {
    const selected = nodes[this.roundRobinIndex % nodes.length];
    this.roundRobinIndex = (this.roundRobinIndex + 1) % nodes.length;
    return selected;
  }

  private selectWeightedNode(nodes: LoadBalancerNode[]): LoadBalancerNode {
    const totalWeight = nodes.reduce((sum, node) => sum + node.weight, 0);

    if (totalWeight === 0) {
      return nodes[0]; // Fallback to first node
    }

    let random = Math.random() * totalWeight;

    for (const node of nodes) {
      random -= node.weight;
      if (random <= 0) {
        return node;
      }
    }

    return nodes[nodes.length - 1]; // Fallback to last node
  }

  private selectLeastLoadedNode(nodes: LoadBalancerNode[]): LoadBalancerNode {
    let leastLoadedNode = nodes[0];
    let minLoad = Number.MAX_VALUE;

    for (const node of nodes) {
      const metrics = this.nodeMetrics.get(node.nodeId);
      const load = metrics?.load ?? 0.5; // Default load if no metrics

      if (load < minLoad) {
        minLoad = load;
        leastLoadedNode = node;
      }
    }

    return leastLoadedNode;
  }

  private selectGeographicNode(nodes: LoadBalancerNode[], request: UncertaintyRequest): LoadBalancerNode {
    // Simple geographic selection - prefer nodes in same region
    // In production, would use client IP geolocation
    const clientRegion = request.clientId?.split('-')[0] || 'us-east-1';

    const localNodes = nodes.filter((node) => node.region === clientRegion);
    if (localNodes.length > 0) {
      return this.selectLeastLoadedNode(localNodes);
    }

    return this.selectLeastLoadedNode(nodes);
  }

  private isNodeHealthy(metrics: NodeMetrics): boolean {
    const maxLatencyMs = 1000;
    const maxErrorRate = 0.05;
    const maxAgeSeconds = 60;

    const now = Date.now();
    const ageSeconds = (now - metrics.lastHeartbeat) / 1000;

    return (
      metrics.latencyMs < maxLatencyMs &&
      metrics.errorRate < maxErrorRate &&
      ageSeconds < maxAgeSeconds
    );
  }

  private startHealthChecking(): void {
    this.healthCheckInterval = setInterval(() => {
      this.performHealthChecks();
    }, this.config.healthCheckInterval);
  }

  private async performHealthChecks(): Promise<void> {
    const now = Date.now();
    const healthCheckPromises: Array<Promise<void>> = [];

    for (const [nodeId, node] of this.nodes) {
      if (now - node.lastHealthCheck > this.config.healthCheckInterval) {
        healthCheckPromises.push(this.checkNodeHealth(nodeId, node));
      }
    }

    await Promise.allSettled(healthCheckPromises);
  }

  private async checkNodeHealth(nodeId: string, node: LoadBalancerNode): Promise<void> {
    try {
      // In production, would make actual HTTP health check
      // For now, simulate based on last known metrics
      const metrics = this.nodeMetrics.get(nodeId);
      const wasHealthy = node.isHealthy;

      if (metrics) {
        node.isHealthy = this.isNodeHealthy(metrics);
      } else {
        // No recent metrics - consider unhealthy
        node.isHealthy = false;
      }

      node.lastHealthCheck = Date.now();

      if (wasHealthy !== node.isHealthy) {
        this.logger.info(`Node health status changed`, {
          nodeId,
          wasHealthy,
          isHealthy: node.isHealthy,
        });
        this.emit('nodeHealthChanged', { nodeId, isHealthy: node.isHealthy });
      }
    } catch (error) {
      this.logger.error('Health check failed for node', { nodeId, error });
      node.isHealthy = false;
      node.consecutiveFailures++;
    }
  }
}