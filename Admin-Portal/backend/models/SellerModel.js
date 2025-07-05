import pool from "../config/db.js";

export const createSeller=async(name, email)=>{
    const result=await pool.query(
        `INSERT INTO sellers (name, email, created_at) 
     VALUES ($1, $2, NOW()) RETURNING *`,
    [name, email]
    );

    return result.rows[0]
}
