/**
 * TypeScript Circuit Breaker Implementation
 *
 * Provides fault tolerance for calls to Python services with
 * configurable thresholds, timeouts, and half-open state testing.
 */

import { EventEmitter } from 'events';
import { Logger } from 'winston';
import { CircuitBreakerConfig } from '../types/calibration';
import { createLogger } from './logger';

export type CircuitBreakerState = 'CLOSED' | 'OPEN' | 'HALF_OPEN';

interface CircuitBreakerMetrics {
  totalCalls: number;
  successfulCalls: number;
  failedCalls: number;
  lastFailureTime: number;
  stateChanges: number;
  avgResponseTime: number;
}

export class CircuitBreaker extends EventEmitter {
  private state: CircuitBreakerState = 'CLOSED';
  private failureCount = 0;
  private lastFailureTime = 0;
  private halfOpenTestTime = 0;
  private logger: Logger;
  private metrics: CircuitBreakerMetrics;

  constructor(private config: CircuitBreakerConfig) {
    super();
    this.logger = createLogger('CircuitBreaker');
    this.metrics = this.initializeMetrics();
  }

  /**
   * Execute a function with circuit breaker protection
   */
  public async execute<T>(fn: () => Promise<T>): Promise<T> {
    const startTime = Date.now();

    // Check circuit breaker state
    this.updateState();

    if (this.state === 'OPEN') {
      this.logger.warn('Circuit breaker is OPEN - rejecting request');
      this.emit('requestRejected', { reason: 'circuit_open' });
      throw new Error('Circuit breaker is OPEN - service unavailable');
    }

    this.metrics.totalCalls++;

    try {
      const result = await fn();

      // Success - reset failure count
      this.onSuccess(Date.now() - startTime);
      return result;

    } catch (error) {
      this.onFailure(error as Error, Date.now() - startTime);
      throw error;
    }
  }

  /**
   * Get current circuit breaker state
   */
  public getState(): CircuitBreakerState {
    this.updateState();
    return this.state;
  }

  /**
   * Get circuit breaker metrics
   */
  public getMetrics(): CircuitBreakerMetrics & {
    state: CircuitBreakerState;
    failureRate: number;
    uptime: number;
  } {
    return {
      ...this.metrics,
      state: this.state,
      failureRate: this.metrics.totalCalls > 0 ?
        this.metrics.failedCalls / this.metrics.totalCalls : 0,
      uptime: Date.now() - this.lastFailureTime,
    };
  }

  /**
   * Force circuit breaker to specific state (for testing)
   */
  public forceState(state: CircuitBreakerState): void {
    const previousState = this.state;
    this.state = state;
    this.metrics.stateChanges++;

    this.logger.info('Circuit breaker state forced', {
      previousState,
      newState: state,
    });

    this.emit('stateChanged', { from: previousState, to: state, forced: true });
  }

  /**
   * Reset circuit breaker to initial state
   */
  public reset(): void {
    this.state = 'CLOSED';
    this.failureCount = 0;
    this.lastFailureTime = 0;
    this.halfOpenTestTime = 0;
    this.metrics = this.initializeMetrics();

    this.logger.info('Circuit breaker reset');
    this.emit('reset');
  }

  // Private methods

  private updateState(): void {
    const now = Date.now();

    switch (this.state) {
      case 'CLOSED':
        // Check if we should open due to failures
        if (this.shouldOpen()) {
          this.transitionTo('OPEN');
        }
        break;

      case 'OPEN':
        // Check if we should try half-open
        if (now - this.lastFailureTime >= this.config.timeout) {
          this.transitionTo('HALF_OPEN');
          this.halfOpenTestTime = now;
        }
        break;

      case 'HALF_OPEN':
        // Half-open has its own logic in success/failure handlers
        // Check timeout for half-open state
        if (now - this.halfOpenTestTime >= this.config.halfOpenTimeout) {
          this.transitionTo('OPEN');
        }
        break;
    }
  }

  private shouldOpen(): boolean {
    // Open if we have enough failures within the monitoring window
    const now = Date.now();
    const windowStart = now - this.config.monitoringWindow;

    // For simplicity, using total failure count
    // In production, would track failures within time window
    return (
      this.failureCount >= this.config.failureThreshold &&
      this.lastFailureTime > windowStart
    );
  }

  private transitionTo(newState: CircuitBreakerState): void {
    const previousState = this.state;

    if (previousState !== newState) {
      this.state = newState;
      this.metrics.stateChanges++;

      this.logger.info('Circuit breaker state changed', {
        from: previousState,
        to: newState,
        failureCount: this.failureCount,
        lastFailureAge: Date.now() - this.lastFailureTime,
      });

      this.emit('stateChanged', { from: previousState, to: newState });

      // Reset failure count when closing
      if (newState === 'CLOSED') {
        this.failureCount = 0;
      }
    }
  }

  private onSuccess(responseTime: number): void {
    this.metrics.successfulCalls++;

    // Update rolling average response time
    const alpha = 0.1;
    this.metrics.avgResponseTime =
      this.metrics.avgResponseTime * (1 - alpha) + responseTime * alpha;

    if (this.state === 'HALF_OPEN') {
      // Success in half-open state - transition to closed
      this.transitionTo('CLOSED');
    }

    // Reset failure count on success
    this.failureCount = 0;

    this.emit('success', { responseTime, state: this.state });
  }

  private onFailure(error: Error, responseTime: number): void {
    this.failureCount++;
    this.metrics.failedCalls++;
    this.lastFailureTime = Date.now();

    this.logger.warn('Circuit breaker recorded failure', {
      error: error.message,
      failureCount: this.failureCount,
      state: this.state,
      responseTime,
    });

    if (this.state === 'HALF_OPEN') {
      // Failure in half-open state - transition back to open
      this.transitionTo('OPEN');
    }

    this.emit('failure', {
      error: error.message,
      failureCount: this.failureCount,
      state: this.state,
      responseTime,
    });
  }

  private initializeMetrics(): CircuitBreakerMetrics {
    return {
      totalCalls: 0,
      successfulCalls: 0,
      failedCalls: 0,
      lastFailureTime: 0,
      stateChanges: 0,
      avgResponseTime: 0,
    };
  }
}