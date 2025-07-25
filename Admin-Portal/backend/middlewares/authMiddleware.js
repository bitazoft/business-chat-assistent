import jwt from "jsonwebtoken"

const verifyToken = (token) => {
    if (!token) {
      throw new Error("No token provided");
    }
  
    try {
        console.log(token)
      return jwt.verify(token, process.env.JWT_SECRET);
    } catch (err) {
      throw new Error("Invalid or expired token");
    }
  };

export default verifyToken;