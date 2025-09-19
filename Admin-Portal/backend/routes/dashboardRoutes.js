import express from 'express';
import {
  recentOrders,
  overview,
  topProducts,
  popularProducts,
  getCustomersWithMoreThanMessages,
  messagesByTimePeriodsForDate,
  messagesByLast7Days,
  messagesByMonths,
  getAvgResponseTime,
  getTotalRevenueByDate,
  getTotalRevenueByMonth,
  getOrdersByDate,
  getOrdersByMonth
} from '../controllers/dashboardController.js';
import authenticate from '../middlewares/authMiddleware.js';
import authorizeRole from '../middlewares/roleMiddleware.js';

const router=express.Router();

// router.get('/overview', authenticate, overview);
// router.get('/recent-orders', authenticate, recentOrders);
// router.get('/top-products', authenticate, topProducts);

router.get('/overview/:id', authenticate, authorizeRole("seller"), overview);
router.get('/recent-orders/:id', authenticate, authorizeRole("seller"), recentOrders);
router.get('/top-products/:id', authenticate, authorizeRole("seller"), topProducts);
router.get('/popular-products/:id', authenticate, authorizeRole("seller"), popularProducts);
router.get('/messages-by-time-periods/:id', authenticate, authorizeRole("seller"), messagesByTimePeriodsForDate);
router.get('/messages-by-last-7-days/:id', authenticate, authorizeRole("seller"), messagesByLast7Days);
router.get('/messages-by-months/:id', authenticate, authorizeRole("seller"), messagesByMonths);
router.get('/customers-with-messages/:id', authenticate, authorizeRole("seller"), getCustomersWithMoreThanMessages);
router.get('/avg-response-time/:id', authenticate, authorizeRole("seller"), getAvgResponseTime);

// New revenue and orders endpoints
router.get('/revenue-by-date/:id', authenticate, authorizeRole("seller"), getTotalRevenueByDate);
router.get('/revenue-by-month/:id', authenticate, authorizeRole("seller"), getTotalRevenueByMonth);
router.get('/orders-by-date/:id', authenticate, authorizeRole("seller"), getOrdersByDate);
router.get('/orders-by-month/:id', authenticate, authorizeRole("seller"), getOrdersByMonth);

export default router;