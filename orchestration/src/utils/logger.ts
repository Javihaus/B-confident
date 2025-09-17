/**
 * Structured Logger Configuration for TypeScript Orchestration Layer
 */

import winston from 'winston';

const logLevel = process.env.LOG_LEVEL || 'info';

export const createLogger = (component: string): winston.Logger => {
  return winston.createLogger({
    level: logLevel,
    format: winston.format.combine(
      winston.format.timestamp(),
      winston.format.errors({ stack: true }),
      winston.format.json(),
      winston.format.printf(({ timestamp, level, message, component: comp, ...meta }) => {
        return JSON.stringify({
          timestamp,
          level,
          component: comp || component,
          message,
          ...meta,
        });
      })
    ),
    transports: [
      new winston.transports.Console({
        format: winston.format.combine(
          winston.format.colorize(),
          winston.format.simple()
        )
      }),
      new winston.transports.File({
        filename: 'logs/orchestration.log',
        format: winston.format.json()
      })
    ],
  });
};