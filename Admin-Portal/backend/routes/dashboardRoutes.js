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

const router=express.Router();

// router.get('/overview', authenticate, overview);
// router.get('/recent-orders', authenticate, recentOrders);
// router.get('/top-products', authenticate, topProducts);

router.get('/overview/:id', overview);
router.get('/recent-orders/:id', recentOrders);
router.get('/top-products/:id', topProducts);
router.get('/popular-products/:id', popularProducts);
router.get('/messages-by-time-periods/:id', messagesByTimePeriodsForDate);
router.get('/messages-by-last-7-days/:id', messagesByLast7Days);
router.get('/messages-by-months/:id', messagesByMonths);
router.get('/customers-with-messages/:id', getCustomersWithMoreThanMessages);
router.get('/avg-response-time/:id', getAvgResponseTime);

// New revenue and orders endpoints
router.get('/revenue-by-date/:id', getTotalRevenueByDate);
router.get('/revenue-by-month/:id', getTotalRevenueByMonth);
router.get('/orders-by-date/:id', getOrdersByDate);
router.get('/orders-by-month/:id', getOrdersByMonth);

export default router;