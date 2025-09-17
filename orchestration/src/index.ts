/**
 * Main Entry Point for B-Confident TypeScript Orchestration Layer
 *
 * Production-ready orchestration service that coordinates distributed
 * uncertainty quantification with auto-scaling, load balancing, and monitoring.
 */

import { config } from 'dotenv';
import { OrchestrationAPI } from './services/OrchestrationAPI';
import { OrchestrationConfig } from './types/calibration';
import { createLogger } from './utils/logger';

// Load environment variables
config();

const logger = createLogger('Main');

/**
 * Load configuration from environment variables with defaults
 */
function loadConfiguration(): OrchestrationConfig {
  return {
    port: parseInt(process.env.PORT || '3000', 10),
    redisUrl: process.env.REDIS_URL || 'redis://localhost:6379',
    pythonServiceUrl: process.env.PYTHON_SERVICE_URL || 'http://localhost:8000',
    clusterId: process.env.CLUSTER_ID || 'default-cluster',
    region: process.env.REGION || 'us-east-1',

    autoScaler: {
      minNodes: parseInt(process.env.AUTO_SCALER_MIN_NODES || '2', 10),
      maxNodes: parseInt(process.env.AUTO_SCALER_MAX_NODES || '20', 10),
      scaleUpThreshold: parseFloat(process.env.AUTO_SCALER_SCALE_UP_THRESHOLD || '0.8'),
      scaleDownThreshold: parseFloat(process.env.AUTO_SCALER_SCALE_DOWN_THRESHOLD || '0.3'),
      cooldownPeriod: parseInt(process.env.AUTO_SCALER_COOLDOWN_MS || '300000', 10), // 5 minutes
      evaluationInterval: parseInt(process.env.AUTO_SCALER_EVAL_INTERVAL_MS || '60000', 10), // 1 minute
    },

    loadBalancer: {
      strategy: (process.env.LOAD_BALANCER_STRATEGY as any) || 'weighted',
      healthCheckInterval: parseInt(process.env.LOAD_BALANCER_HEALTH_CHECK_MS || '30000', 10),
      maxRetries: parseInt(process.env.LOAD_BALANCER_MAX_RETRIES || '3', 10),
      timeoutMs: parseInt(process.env.LOAD_BALANCER_TIMEOUT_MS || '30000', 10),
    },

    circuitBreaker: {
      failureThreshold: parseInt(process.env.CIRCUIT_BREAKER_FAILURE_THRESHOLD || '5', 10),
      timeout: parseInt(process.env.CIRCUIT_BREAKER_TIMEOUT_MS || '60000', 10), // 1 minute
      halfOpenTimeout: parseInt(process.env.CIRCUIT_BREAKER_HALF_OPEN_MS || '30000', 10), // 30 seconds
      monitoringWindow: parseInt(process.env.CIRCUIT_BREAKER_WINDOW_MS || '300000', 10), // 5 minutes
    },

    enableMetrics: process.env.ENABLE_METRICS !== 'false',
    enableDashboard: process.env.ENABLE_DASHBOARD !== 'false',
    logLevel: (process.env.LOG_LEVEL as any) || 'info',
  };
}

/**
 * Main application startup
 */
async function main(): Promise<void> {
  try {
    logger.info('Starting B-Confident Orchestration Layer...', {
      nodeVersion: process.version,
      platform: process.platform,
      arch: process.arch,
    });

    // Load configuration
    const config = loadConfiguration();
    logger.info('Configuration loaded', {
      port: config.port,
      clusterId: config.clusterId,
      region: config.region,
      autoScaler: {
        minNodes: config.autoScaler.minNodes,
        maxNodes: config.autoScaler.maxNodes,
      },
      loadBalancer: {
        strategy: config.loadBalancer.strategy,
      },
    });

    // Initialize orchestration API
    const orchestrationAPI = new OrchestrationAPI(config);

    // Setup graceful shutdown
    const shutdown = async (signal: string) => {
      logger.info(`Received ${signal} - starting graceful shutdown...`);

      try {
        await orchestrationAPI.shutdown();
        logger.info('Graceful shutdown completed');
        process.exit(0);
      } catch (error) {
        logger.error('Error during shutdown', { error });
        process.exit(1);
      }
    };

    // Handle shutdown signals
    process.on('SIGTERM', () => shutdown('SIGTERM'));
    process.on('SIGINT', () => shutdown('SIGINT'));

    // Handle uncaught exceptions
    process.on('uncaughtException', (error) => {
      logger.error('Uncaught exception', { error: error.message, stack: error.stack });
      process.exit(1);
    });

    process.on('unhandledRejection', (reason, promise) => {
      logger.error('Unhandled rejection', { reason, promise });
      process.exit(1);
    });

    // Start the orchestration API
    await orchestrationAPI.start();

    logger.info('B-Confident Orchestration Layer started successfully', {
      port: config.port,
      clusterId: config.clusterId,
      healthEndpoint: `http://localhost:${config.port}/health`,
      dashboardEndpoint: `http://localhost:${config.port}/dashboard/metrics`,
    });

  } catch (error) {
    logger.error('Failed to start Orchestration Layer', { error });
    process.exit(1);
  }
}

// Start the application
if (require.main === module) {
  main().catch((error) => {
    console.error('Fatal startup error:', error);
    process.exit(1);
  });
}

export { main, loadConfiguration };