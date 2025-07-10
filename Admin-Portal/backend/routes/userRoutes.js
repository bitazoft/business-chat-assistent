import express from "express"
import verifyToken from "../middlewares/authMiddleware.js";
import authorizeRole from "../middlewares/roleMiddleware.js";

const router = express.Router();

router.get("/admin", verifyToken, authorizeRole("admin"), (req, res) => {
    res.json({message: "Welcome admin"})
})

router.get("/manager", verifyToken, authorizeRole("admin", "seller"), (req, res) => {
    res.json({message: "Welcome manager"})
})

router.get("/user", verifyToken, authorizeRole("admin", "seller", "customer"), (req, res) => {
    res.json({message: "Welcome user"})
})

export default router;