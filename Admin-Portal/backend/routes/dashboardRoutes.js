import express from 'express';
import {recentOrders,overview,topProducts,popularProducts,getCustomersWithMoreThanMessages,messagesByTimePeriodsForDate, getAvgResponseTime } from '../controllers/dashboardController.js';
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
router.get('/customers-with-messages/:id', getCustomersWithMoreThanMessages);
router.get('/avg-response-time/:id', getAvgResponseTime);

export default router;