import express from 'express';
import { createOrder, deleteOrder, getAllOrders, updateOrder, updateOrderStatus, addOrderItem, removeOrderItem } from '../controllers/orderController.js';
import authenticate from '../middlewares/authMiddleware.js';
import authorizeRole from '../middlewares/roleMiddleware.js';

const router = express.Router();

router.get('/:seller_id', authenticate, authorizeRole("seller"), getAllOrders);
// router.get('/:id', authenticate, getOrderById);
router.post('/', authenticate, authorizeRole("seller"), createOrder);
router.put('/:id', authenticate, authorizeRole("seller"), updateOrder);
router.delete('/:id', authenticate, authorizeRole("seller"), deleteOrder);

// Order items management
router.post('/:id/items', authenticate, authorizeRole("seller"), addOrderItem);
router.delete('/:id/items/:itemId', authenticate, authorizeRole("seller"), removeOrderItem);

export default router;
