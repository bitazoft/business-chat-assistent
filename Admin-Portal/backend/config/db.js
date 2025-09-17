// config/db.js
import dotenv from "dotenv";
import { Pool } from "pg";

dotenv.config();

const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  ssl: {
    rejectUnauthorized: false
  }
});

const dbConnect = async () => {
  try {
    const client = await pool.connect();
    const { host, database } = client.connectionParameters;
    console.log(`✅ Connected to PostgreSQL`);
    console.log(`🔹 Host: ${host}`);
    console.log(`🔹 Database: ${database}`);
    client.release(); 
  } catch (err) {
    console.error("❌ PostgreSQL connection error:", err.message);
    process.exit(1);
  }
};

export { dbConnect, pool };
