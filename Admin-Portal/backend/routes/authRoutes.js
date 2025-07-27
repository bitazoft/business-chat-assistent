import express from 'express';
import { login, logout, register, checkAuth } from '../controllers/authController.js';

const router=express.Router();

router.post('/register', register);
router.post('/login', login);
router.post('/logout', logout);
router.get('/me', checkAuth);

export default router;