import bcrypt from "bcryptjs";
import jwt from "jsonwebtoken";
import userService from "../services/userService.js";
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

    const isMatch = bcrypt.compare(password, user.password);

    if (!isMatch) {
      return res.status(400).json({ message: `Password is incorrect..` });
    }

    const token = jwt.sign(
      { id: user.id, role: user.role },
      process.env.JWT_SECRET,
      { expiresIn: "1h" }
    );

    res.status(200).json({ token , user});
  } catch (err) {
    res.status(500).json({error: err.message})
  }
};

export { register, login };
