import express from 'express';
import { getAllUsers } from '../controllers/userController.js';
import authenticate from '../middlewares/authMiddleware.js';
import authorizeRole from '../middlewares/roleMiddleware.js';

const router = express.Router();

router.get('/getAllUsers', getAllUsers);

export default router;