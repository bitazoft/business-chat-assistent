import express from 'express';
import { createOrder, deleteOrder, getAllOrders, updateOrder, updateOrderStatus, addOrderItem, removeOrderItem } from '../controllers/orderController.js';
import authenticate from '../middlewares/authMiddleware.js';

const router = express.Router();

router.get('/:seller_id', authenticate, getAllOrders);
// router.get('/:id', authenticate, getOrderById);
router.post('/', authenticate, createOrder);
router.put('/:id', authenticate, updateOrder);
router.delete('/:id', authenticate, deleteOrder);

// Order items management
router.post('/:id/items', authenticate, addOrderItem);
router.delete('/:id/items/:itemId', authenticate, removeOrderItem);

export default router;
