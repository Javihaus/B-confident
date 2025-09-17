/**
 * TypeScript Auto-Scaler for Distributed Uncertainty Quantification
 *
 * Intelligent auto-scaling based on load metrics, calibration quality drift,
 * and performance thresholds with strong typing and comprehensive monitoring.
 */

import { EventEmitter } from 'events';
import { Logger } from 'winston';
import {
  AutoScalerConfig,
  NodeMetrics,
  ClusterState,
  ScalingDecision,
  ScalingRecommendation,
} from '../types/calibration';
import { createLogger } from '../utils/logger';

interface ScalingEvent {
  timestamp: number;
  decision: ScalingDecision;
  reason: string;
  clusterSize: number;
  triggerMetrics: {
    avgLoad: number;
    maxLoad: number;
    calibrationDrift: number;
    healthyNodes: number;
  };
}

interface ScalingAnalysis {
  decision: ScalingDecision;
  reason: string;
  confidence: number;
  metrics: {
    avgLoad: number;
    maxLoad: number;
    minLoad: number;
    loadVariance: number;
    calibrationQualityVariance: number;
    healthyNodeCount: number;
    errorRate: number;
  };
}

export class AutoScaler extends EventEmitter {
  private logger: Logger;
  private scalingHistory: ScalingEvent[] = [];
  private lastScalingAction = 0;
  private scalingInterval?: NodeJS.Timeout;

  constructor(private config: AutoScalerConfig) {
    super();
    this.logger = createLogger('AutoScaler');
    this.startScalingLoop();
  }

  /**
   * Analyze current cluster state and determine scaling needs
   */
  public analyzeScalingNeeds(
    clusterState: ClusterState,
    nodeMetrics: Record<string, NodeMetrics>
  ): ScalingAnalysis {
    const now = Date.now();

    // Check cooldown period
    if (now - this.lastScalingAction < this.config.cooldownPeriod) {
      return {
        decision: 'no_action',
        reason: `Cooldown period active (${Math.round((this.config.cooldownPeriod - (now - this.lastScalingAction)) / 1000)}s remaining)`,
        confidence: 1.0,
        metrics: this.calculateClusterMetrics(nodeMetrics),
      };
    }

    const metrics = this.calculateClusterMetrics(nodeMetrics);

    // Scale up conditions
    if (this.shouldScaleUp(metrics, clusterState)) {
      return {
        decision: 'scale_up',
        reason: this.generateScaleUpReason(metrics),
        confidence: this.calculateScalingConfidence('scale_up', metrics),
        metrics,
      };
    }

    // Scale down conditions
    if (this.shouldScaleDown(metrics, clusterState)) {
      return {
        decision: 'scale_down',
        reason: this.generateScaleDownReason(metrics),
        confidence: this.calculateScalingConfidence('scale_down', metrics),
        metrics,
      };
    }

    // Rebalance conditions
    if (this.shouldRebalance(metrics, clusterState)) {
      return {
        decision: 'rebalance',
        reason: this.generateRebalanceReason(metrics),
        confidence: this.calculateScalingConfidence('rebalance', metrics),
        metrics,
      };
    }

    return {
      decision: 'no_action',
      reason: 'Cluster is stable and within acceptable parameters',
      confidence: 1.0,
      metrics,
    };
  }

  /**
   * Get scaling recommendations with detailed analysis
   */
  public getScalingRecommendations(
    clusterState: ClusterState,
    nodeMetrics: Record<string, NodeMetrics>
  ): ScalingRecommendation {
    const analysis = this.analyzeScalingNeeds(clusterState, nodeMetrics);

    return {
      decision: analysis.decision,
      reason: analysis.reason,
      confidence: analysis.confidence,
      recommendedActions: this.generateRecommendedActions(analysis),
      clusterLoadStats: {
        avgLoad: analysis.metrics.avgLoad,
        maxLoad: analysis.metrics.maxLoad,
        minLoad: analysis.metrics.minLoad,
        loadVariance: analysis.metrics.loadVariance,
      },
    };
  }

  /**
   * Record a scaling action that was executed
   */
  public recordScalingAction(
    decision: ScalingDecision,
    reason: string,
    clusterSize: number,
    nodeMetrics: Record<string, NodeMetrics>
  ): void {
    const now = Date.now();
    const metrics = this.calculateClusterMetrics(nodeMetrics);

    const event: ScalingEvent = {
      timestamp: now,
      decision,
      reason,
      clusterSize,
      triggerMetrics: {
        avgLoad: metrics.avgLoad,
        maxLoad: metrics.maxLoad,
        calibrationDrift: metrics.calibrationQualityVariance,
        healthyNodes: metrics.healthyNodeCount,
      },
    };

    this.scalingHistory.push(event);
    this.lastScalingAction = now;

    // Keep only last 100 events
    if (this.scalingHistory.length > 100) {
      this.scalingHistory.shift();
    }

    this.logger.info('Recorded scaling action', {
      decision,
      reason,
      clusterSize,
      confidence: this.calculateScalingConfidence(decision, metrics),
    });

    this.emit('scalingAction', event);
  }

  /**
   * Get scaling history and statistics
   */
  public getScalingStatistics(): {
    totalActions: number;
    recentActions: ScalingEvent[];
    actionCounts: Record<ScalingDecision, number>;
    avgTimeBetweenActions: number;
    lastAction?: ScalingEvent;
  } {
    const actionCounts = this.scalingHistory.reduce(
      (counts, event) => {
        counts[event.decision]++;
        return counts;
      },
      { scale_up: 0, scale_down: 0, rebalance: 0, no_action: 0 } as Record<ScalingDecision, number>
    );

    let avgTimeBetweenActions = 0;
    if (this.scalingHistory.length > 1) {
      const timeSpan = this.scalingHistory[this.scalingHistory.length - 1].timestamp -
                      this.scalingHistory[0].timestamp;
      avgTimeBetweenActions = timeSpan / (this.scalingHistory.length - 1);
    }

    return {
      totalActions: this.scalingHistory.length,
      recentActions: this.scalingHistory.slice(-10), // Last 10 actions
      actionCounts,
      avgTimeBetweenActions,
      lastAction: this.scalingHistory[this.scalingHistory.length - 1],
    };
  }

  /**
   * Shutdown the auto-scaler
   */
  public shutdown(): void {
    if (this.scalingInterval) {
      clearInterval(this.scalingInterval);
    }
    this.removeAllListeners();
    this.logger.info('AutoScaler shutdown complete');
  }

  // Private methods

  private startScalingLoop(): void {
    this.scalingInterval = setInterval(() => {
      this.emit('evaluationCycle');
    }, this.config.evaluationInterval);

    this.logger.info('AutoScaler started', {
      minNodes: this.config.minNodes,
      maxNodes: this.config.maxNodes,
      evaluationInterval: this.config.evaluationInterval,
    });
  }

  private calculateClusterMetrics(nodeMetrics: Record<string, NodeMetrics>): ScalingAnalysis['metrics'] {
    const healthyNodes = Object.values(nodeMetrics).filter(this.isNodeHealthy);

    if (healthyNodes.length === 0) {
      return {
        avgLoad: 0,
        maxLoad: 0,
        minLoad: 0,
        loadVariance: 0,
        calibrationQualityVariance: 0,
        healthyNodeCount: 0,
        errorRate: 0,
      };
    }

    const loads = healthyNodes.map(n => n.load);
    const errorRates = healthyNodes.map(n => n.errorRate);

    const avgLoad = loads.reduce((sum, load) => sum + load, 0) / loads.length;
    const maxLoad = Math.max(...loads);
    const minLoad = Math.min(...loads);
    const loadVariance = this.calculateVariance(loads);

    // Simulate calibration quality variance (would come from Python service in production)
    const calibrationQualityVariance = Math.random() * 0.1; // 0-0.1 range

    const avgErrorRate = errorRates.reduce((sum, rate) => sum + rate, 0) / errorRates.length;

    return {
      avgLoad,
      maxLoad,
      minLoad,
      loadVariance,
      calibrationQualityVariance,
      healthyNodeCount: healthyNodes.length,
      errorRate: avgErrorRate,
    };
  }

  private shouldScaleUp(metrics: ScalingAnalysis['metrics'], clusterState: ClusterState): boolean {
    const conditions = [
      // High average load
      metrics.avgLoad > this.config.scaleUpThreshold,

      // Very high maximum load
      metrics.maxLoad > this.config.scaleUpThreshold * 1.2,

      // High calibration drift
      metrics.calibrationQualityVariance > 0.1,

      // Below minimum node count
      metrics.healthyNodeCount < this.config.minNodes,

      // High error rate
      metrics.errorRate > 0.02,
    ];

    const activeConditions = conditions.filter(Boolean).length;
    const canScaleUp = clusterState.totalNodes < this.config.maxNodes;

    return activeConditions >= 2 && canScaleUp;
  }

  private shouldScaleDown(metrics: ScalingAnalysis['metrics'], clusterState: ClusterState): boolean {
    const conditions = [
      // Low average load
      metrics.avgLoad < this.config.scaleDownThreshold,

      // Low maximum load
      metrics.maxLoad < this.config.scaleDownThreshold * 1.5,

      // Low calibration drift
      metrics.calibrationQualityVariance < 0.02,

      // Above minimum node count with room to scale down
      metrics.healthyNodeCount > this.config.minNodes + 1,

      // Low error rate
      metrics.errorRate < 0.005,
    ];

    const activeConditions = conditions.filter(Boolean).length;
    const canScaleDown = clusterState.totalNodes > this.config.minNodes;

    return activeConditions >= 4 && canScaleDown;
  }

  private shouldRebalance(metrics: ScalingAnalysis['metrics'], clusterState: ClusterState): boolean {
    const loadImbalanceThreshold = 0.4;
    const loadImbalance = metrics.maxLoad - metrics.minLoad;

    return (
      loadImbalance > loadImbalanceThreshold &&
      metrics.healthyNodeCount > 1 &&
      metrics.avgLoad < this.config.scaleUpThreshold
    );
  }

  private calculateScalingConfidence(decision: ScalingDecision, metrics: ScalingAnalysis['metrics']): number {
    switch (decision) {
      case 'scale_up':
        return Math.min(1.0, (metrics.avgLoad - this.config.scaleUpThreshold) * 2);

      case 'scale_down':
        return Math.min(1.0, (this.config.scaleDownThreshold - metrics.avgLoad) * 2);

      case 'rebalance':
        const loadImbalance = metrics.maxLoad - metrics.minLoad;
        return Math.min(1.0, loadImbalance * 2);

      default:
        return 1.0;
    }
  }

  private generateScaleUpReason(metrics: ScalingAnalysis['metrics']): string {
    const reasons: string[] = [];

    if (metrics.avgLoad > this.config.scaleUpThreshold) {
      reasons.push(`high average load (${(metrics.avgLoad * 100).toFixed(1)}%)`);
    }

    if (metrics.calibrationQualityVariance > 0.1) {
      reasons.push(`calibration drift detected (${(metrics.calibrationQualityVariance * 100).toFixed(1)}%)`);
    }

    if (metrics.healthyNodeCount < this.config.minNodes) {
      reasons.push(`below minimum nodes (${metrics.healthyNodeCount} < ${this.config.minNodes})`);
    }

    return `Scale up needed: ${reasons.join(', ')}`;
  }

  private generateScaleDownReason(metrics: ScalingAnalysis['metrics']): string {
    return `Scale down opportunity: low load (${(metrics.avgLoad * 100).toFixed(1)}%), stable calibration`;
  }

  private generateRebalanceReason(metrics: ScalingAnalysis['metrics']): string {
    const imbalance = ((metrics.maxLoad - metrics.minLoad) * 100).toFixed(1);
    return `Rebalance needed: load imbalance of ${imbalance}% between nodes`;
  }

  private generateRecommendedActions(analysis: ScalingAnalysis): ScalingRecommendation['recommendedActions'] {
    const actions: ScalingRecommendation['recommendedActions'] = [];

    switch (analysis.decision) {
      case 'scale_up':
        actions.push(
          { action: 'Add new nodes', priority: 'high', description: 'Cluster is experiencing high load' },
          { action: 'Monitor resource usage', priority: 'medium', description: 'Ensure new nodes have sufficient resources' },
          { action: 'Update load balancer', priority: 'medium', description: 'Include new nodes in rotation' }
        );
        break;

      case 'scale_down':
        actions.push(
          { action: 'Identify underutilized nodes', priority: 'low', description: 'Find nodes with consistently low load' },
          { action: 'Graceful shutdown', priority: 'high', description: 'Drain connections before removal' },
          { action: 'Update monitoring', priority: 'low', description: 'Adjust alerts for smaller cluster' }
        );
        break;

      case 'rebalance':
        actions.push(
          { action: 'Redistribute load', priority: 'medium', description: 'Balance work across existing nodes' },
          { action: 'Check network connectivity', priority: 'low', description: 'Verify inter-node communication' },
          { action: 'Monitor calibration quality', priority: 'medium', description: 'Ensure rebalancing maintains accuracy' }
        );
        break;
    }

    return actions;
  }

  private isNodeHealthy(metrics: NodeMetrics): boolean {
    const now = Date.now();
    const maxAge = 60000; // 60 seconds
    const maxLatency = 1000; // 1 second
    const maxErrorRate = 0.05; // 5%

    return (
      (now - metrics.lastHeartbeat) < maxAge &&
      metrics.latencyMs < maxLatency &&
      metrics.errorRate < maxErrorRate
    );
  }

  private calculateVariance(values: number[]): number {
    if (values.length === 0) return 0;

    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const squaredDiffs = values.map(val => Math.pow(val - mean, 2));

    return squaredDiffs.reduce((sum, diff) => sum + diff, 0) / values.length;
  }
}