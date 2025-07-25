import bcrypt from "bcryptjs";
import jwt from "jsonwebtoken";
import userService from "../services/userService.js";
import verifyToken from "../middlewares/authMiddleware.js";
import { error } from "console";

const register = async (req, res) => {
  try {
    const { name, email, password, phone, address, role } = req.body;
    const hashedPassword = await bcrypt.hash(password, 12);

    const user = await userService.createUser({
      name,
      email,
      phone,
      password: hashedPassword,
      role,
      address,
    });

    res.status(201).json({ message: `User registered !! ${user.email}` });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Registration failed" });
  }
};

const login = async (req, res) => {
  try {
    const { email, password } = req.body;
    const user = await userService.getUserByEmail(email);

    if (!user) {
      return res
        .status(404)
        .json({ message: `User not found with ${email} this email.` });
    }

    const isMatch = await bcrypt.compare(password, user.password);

    if (!isMatch) {
      return res.status(400).json({ message: `Password is incorrect..` });
    }

    const token = jwt.sign(
      { id: user.id, role: user.role },
      process.env.JWT_SECRET,
      { expiresIn: "1h" }
    );

    res.cookie("access_token", token, {
      httpOnly: true,
      secure: process.env.NODE_ENV === "production", 
      sameSite: "strict",
      maxAge:  3600000, // 1 hour in milliseconds
    });

    res.status(200).json({
      message: "Login successful",
      user: {
        id: user.id,
        name: user.name,
        email: user.email,
        role: user.role,
      },
    });
  } catch (err) {
    res.status(500).json({error: err.message})
  }
};

const logout = async (req, res) => {
  res.clearCookie("access_token", {
    httpOnly: true,
    sameSite: "strict",
    secure: process.env.NODE_ENV === "production",
  });
  return res.status(200).json({ message: "Logged out successfully" });
};

const checkAuth = (req, res) => {
  try {
    const token = req.cookies.access_token;
    const decoded = verifyToken(token);
    
    res.status(200).json({ user: decoded });
  } catch (err) {
    res.status(401).json({ message: err.message });
  }
}

export { register, login, logout, checkAuth };
