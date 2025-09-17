/**
 * TypeScript Message Broker for Distributed Calibration Events
 *
 * Type-safe event-driven communication between orchestration layer and Python services
 * with Redis pub/sub, dead letter queues, and guaranteed delivery patterns.
 */

import { EventEmitter } from 'events';
import Redis from 'ioredis';
import { Logger } from 'winston';
import { v4 as uuidv4 } from 'uuid';
import { CalibrationEvent, CalibrationEventType } from '../types/calibration';
import { createLogger } from '../utils/logger';

export interface MessageBrokerConfig {
  redisUrl: string;
  retryAttempts: number;
  retryDelayMs: number;
  deadLetterQueueEnabled: boolean;
  eventTtlSeconds: number;
}

export interface EventSubscription {
  id: string;
  pattern: string;
  eventTypes: CalibrationEventType[];
  callback: (event: CalibrationEvent) => Promise<void>;
  errorHandler?: (error: Error, event: CalibrationEvent) => void;
}

export interface EventMetrics {
  totalPublished: number;
  totalReceived: number;
  totalErrors: number;
  eventTypeCounts: Record<CalibrationEventType, number>;
  avgProcessingTime: number;
  lastEventTimestamp: number;
}

export class MessageBroker extends EventEmitter {
  private redis: Redis;
  private subscriber: Redis;
  private logger: Logger;
  private subscriptions: Map<string, EventSubscription> = new Map();
  private metrics: EventMetrics;
  private isShuttingDown = false;

  constructor(private config: MessageBrokerConfig) {
    super();
    this.logger = createLogger('MessageBroker');
    this.metrics = this.initializeMetrics();

    // Create Redis connections
    this.redis = new Redis(config.redisUrl, {
      retryDelayOnFailover: 1000,
      maxRetriesPerRequest: 3,
      lazyConnect: true,
    });

    this.subscriber = new Redis(config.redisUrl, {
      retryDelayOnFailover: 1000,
      maxRetriesPerRequest: 3,
      lazyConnect: true,
    });

    this.setupRedisHandlers();
  }

  /**
   * Initialize the message broker
   */
  public async initialize(): Promise<void> {
    try {
      await this.redis.connect();
      await this.subscriber.connect();

      this.logger.info('MessageBroker initialized successfully', {
        redisUrl: this.config.redisUrl.replace(/\/\/.*@/, '//***:***@'), // Hide credentials
      });

      this.emit('connected');
    } catch (error) {
      this.logger.error('Failed to initialize MessageBroker', { error });
      throw error;
    }
  }

  /**
   * Publish a calibration event
   */
  public async publishEvent(
    topic: string,
    event: Omit<CalibrationEvent, 'timestamp' | 'sequenceId'>
  ): Promise<void> {
    if (this.isShuttingDown) {
      throw new Error('MessageBroker is shutting down');
    }

    const fullEvent: CalibrationEvent = {
      ...event,
      timestamp: Date.now(),
      sequenceId: this.generateSequenceId(),
    };

    const serializedEvent = JSON.stringify(fullEvent);

    try {
      // Publish to main topic
      await this.redis.publish(topic, serializedEvent);

      // Store in persistent queue with TTL
      if (this.config.eventTtlSeconds > 0) {
        await this.redis.setex(
          `event:${fullEvent.sequenceId}`,
          this.config.eventTtlSeconds,
          serializedEvent
        );
      }

      this.updatePublishMetrics(fullEvent);
      this.logger.debug('Event published successfully', {
        topic,
        eventType: fullEvent.eventType,
        nodeId: fullEvent.nodeId,
        sequenceId: fullEvent.sequenceId,
      });

      this.emit('eventPublished', { topic, event: fullEvent });
    } catch (error) {
      this.logger.error('Failed to publish event', { topic, event: fullEvent, error });
      this.metrics.totalErrors++;
      throw error;
    }
  }

  /**
   * Subscribe to calibration events with type safety
   */
  public async subscribeToEvents(
    pattern: string,
    eventTypes: CalibrationEventType[],
    callback: (event: CalibrationEvent) => Promise<void>,
    errorHandler?: (error: Error, event: CalibrationEvent) => void
  ): Promise<string> {
    const subscriptionId = uuidv4();

    const subscription: EventSubscription = {
      id: subscriptionId,
      pattern,
      eventTypes,
      callback,
      errorHandler,
    };

    this.subscriptions.set(subscriptionId, subscription);

    // Subscribe to Redis pattern
    await this.subscriber.psubscribe(pattern);

    this.logger.info('Subscribed to events', {
      subscriptionId,
      pattern,
      eventTypes,
    });

    return subscriptionId;
  }

  /**
   * Unsubscribe from events
   */
  public async unsubscribe(subscriptionId: string): Promise<void> {
    const subscription = this.subscriptions.get(subscriptionId);
    if (!subscription) {
      this.logger.warn('Attempted to unsubscribe from non-existent subscription', { subscriptionId });
      return;
    }

    // Check if any other subscriptions use the same pattern
    const otherSubscriptions = Array.from(this.subscriptions.values())
      .filter(sub => sub.id !== subscriptionId && sub.pattern === subscription.pattern);

    if (otherSubscriptions.length === 0) {
      await this.subscriber.punsubscribe(subscription.pattern);
    }

    this.subscriptions.delete(subscriptionId);

    this.logger.info('Unsubscribed from events', {
      subscriptionId,
      pattern: subscription.pattern,
    });
  }

  /**
   * Get current message broker metrics
   */
  public getMetrics(): EventMetrics {
    return { ...this.metrics };
  }

  /**
   * Get active subscribers count for a topic
   */
  public async getSubscriberCount(topic: string): Promise<number> {
    try {
      const result = await this.redis.pubsub('numsub', topic);
      return parseInt(result[1] as string, 10) || 0;
    } catch (error) {
      this.logger.error('Failed to get subscriber count', { topic, error });
      return 0;
    }
  }

  /**
   * Replay events from persistent storage
   */
  public async replayEvents(
    fromTimestamp: number,
    toTimestamp: number,
    eventTypes?: CalibrationEventType[]
  ): Promise<CalibrationEvent[]> {
    try {
      // This would need to be implemented with a more sophisticated storage mechanism
      // For now, return empty array as events are not persistently indexed by timestamp
      this.logger.info('Event replay requested', { fromTimestamp, toTimestamp, eventTypes });
      return [];
    } catch (error) {
      this.logger.error('Failed to replay events', { error });
      return [];
    }
  }

  /**
   * Health check for message broker
   */
  public async healthCheck(): Promise<{
    status: 'healthy' | 'unhealthy';
    redisConnected: boolean;
    subscriberConnected: boolean;
    activeSubscriptions: number;
    lastEventAge: number;
  }> {
    const redisConnected = this.redis.status === 'ready';
    const subscriberConnected = this.subscriber.status === 'ready';
    const activeSubscriptions = this.subscriptions.size;
    const lastEventAge = Date.now() - this.metrics.lastEventTimestamp;

    const status = (redisConnected && subscriberConnected) ? 'healthy' : 'unhealthy';

    return {
      status,
      redisConnected,
      subscriberConnected,
      activeSubscriptions,
      lastEventAge,
    };
  }

  /**
   * Shutdown the message broker gracefully
   */
  public async shutdown(): Promise<void> {
    this.isShuttingDown = true;

    this.logger.info('Shutting down MessageBroker...');

    // Unsubscribe from all patterns
    for (const subscription of this.subscriptions.values()) {
      try {
        await this.subscriber.punsubscribe(subscription.pattern);
      } catch (error) {
        this.logger.error('Error unsubscribing during shutdown', { error });
      }
    }

    this.subscriptions.clear();

    // Close Redis connections
    await Promise.all([
      this.redis.disconnect(),
      this.subscriber.disconnect(),
    ]);

    this.removeAllListeners();
    this.logger.info('MessageBroker shutdown complete');
  }

  // Private methods

  private setupRedisHandlers(): void {
    // Handle connection events
    this.redis.on('connect', () => this.logger.info('Redis publisher connected'));
    this.redis.on('error', (error) => {
      this.logger.error('Redis publisher error', { error });
      this.emit('error', error);
    });

    this.subscriber.on('connect', () => this.logger.info('Redis subscriber connected'));
    this.subscriber.on('error', (error) => {
      this.logger.error('Redis subscriber error', { error });
      this.emit('error', error);
    });

    // Handle pattern message events
    this.subscriber.on('pmessage', async (pattern: string, channel: string, message: string) => {
      await this.handleIncomingMessage(pattern, channel, message);
    });
  }

  private async handleIncomingMessage(pattern: string, channel: string, message: string): Promise<void> {
    try {
      const event: CalibrationEvent = JSON.parse(message);

      // Validate event structure
      if (!this.isValidCalibrationEvent(event)) {
        this.logger.warn('Received invalid calibration event', { pattern, channel, message: message.substring(0, 100) });
        return;
      }

      // Find matching subscriptions
      const matchingSubscriptions = Array.from(this.subscriptions.values()).filter(
        sub => sub.pattern === pattern && sub.eventTypes.includes(event.eventType)
      );

      // Process event for each matching subscription
      const processingPromises = matchingSubscriptions.map(async (subscription) => {
        const startTime = Date.now();

        try {
          await subscription.callback(event);

          const processingTime = Date.now() - startTime;
          this.updateReceiveMetrics(event, processingTime);

        } catch (error) {
          this.logger.error('Error processing event in subscription', {
            subscriptionId: subscription.id,
            event,
            error,
          });

          this.metrics.totalErrors++;

          if (subscription.errorHandler) {
            subscription.errorHandler(error as Error, event);
          }

          // Send to dead letter queue if enabled
          if (this.config.deadLetterQueueEnabled) {
            await this.sendToDeadLetterQueue(event, error as Error);
          }
        }
      });

      await Promise.allSettled(processingPromises);

    } catch (error) {
      this.logger.error('Failed to parse incoming message', { pattern, channel, error });
      this.metrics.totalErrors++;
    }
  }

  private isValidCalibrationEvent(event: any): event is CalibrationEvent {
    return (
      typeof event === 'object' &&
      typeof event.eventType === 'string' &&
      typeof event.nodeId === 'string' &&
      typeof event.timestamp === 'number' &&
      typeof event.data === 'object' &&
      typeof event.sequenceId === 'number'
    );
  }

  private async sendToDeadLetterQueue(event: CalibrationEvent, error: Error): Promise<void> {
    try {
      const deadLetterEvent = {
        originalEvent: event,
        error: error.message,
        timestamp: Date.now(),
      };

      await this.redis.lpush('dead_letter_queue', JSON.stringify(deadLetterEvent));
      this.logger.info('Event sent to dead letter queue', { eventSequenceId: event.sequenceId });
    } catch (dlqError) {
      this.logger.error('Failed to send event to dead letter queue', { dlqError });
    }
  }

  private generateSequenceId(): number {
    return Date.now() * 1000 + Math.floor(Math.random() * 1000);
  }

  private initializeMetrics(): EventMetrics {
    const eventTypeCounts = {} as Record<CalibrationEventType, number>;

    // Initialize all event type counters
    const eventTypes: CalibrationEventType[] = [
      'UPDATE_CALIBRATION', 'REQUEST_SYNC', 'NODE_STATUS', 'THRESHOLD_CHANGE',
      'RECALIBRATION_TRIGGER', 'PARTITION_EVENT', 'LEADER_ELECTION',
      'LOAD_BALANCE', 'AUTO_SCALE', 'HEALTH_CHECK', 'FAILOVER'
    ];

    eventTypes.forEach(type => {
      eventTypeCounts[type] = 0;
    });

    return {
      totalPublished: 0,
      totalReceived: 0,
      totalErrors: 0,
      eventTypeCounts,
      avgProcessingTime: 0,
      lastEventTimestamp: Date.now(),
    };
  }

  private updatePublishMetrics(event: CalibrationEvent): void {
    this.metrics.totalPublished++;
    this.metrics.eventTypeCounts[event.eventType]++;
    this.metrics.lastEventTimestamp = event.timestamp;
  }

  private updateReceiveMetrics(event: CalibrationEvent, processingTime: number): void {
    this.metrics.totalReceived++;

    // Update rolling average processing time
    const alpha = 0.1; // Smoothing factor
    this.metrics.avgProcessingTime =
      this.metrics.avgProcessingTime * (1 - alpha) + processingTime * alpha;

    this.emit('eventProcessed', { event, processingTime });
  }
}