import { createSeller } from "../models/SellerModel.js";

export const registerSeller=async(req,res)=>{
    const { businessName, email}=req.body;

    try {
         const seller=await createSeller(businessName, email);
          res.status(201).json({ success: true, seller });
    } catch (error) {
        console.error('Error registering seller:', error);
    res.status(500).json({ error: 'Failed to register seller', details: error.message
    });
   
}
}