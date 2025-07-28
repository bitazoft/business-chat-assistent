import express from 'express';
import { getAllOrders } from '../controllers/orderController.js';
import authenticate from '../middlewares/authMiddleware.js';

const router=express.Router();

router.get('/getAll/:id', authenticate, getAllOrders);

export default router;