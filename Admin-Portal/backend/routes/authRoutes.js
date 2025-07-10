import express from 'express';
import { login, register } from '../controllers/authController.js';
import verifyToken from "../middlewares/authMiddleware.js";

const router=express.Router();

router.post('/register', register);
router.post('/login',verifyToken, login);

export default router;