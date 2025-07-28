import orderService from "../services/orderService.js";

const getAllOrders = async (req, res) => {
    try {
        const seller_id = parseInt(req.params.id);
        const orders = await orderService.getAllOrders(seller_id);

        res.status(200).json({
            message: `All orders fetched successfully !!`,
            orders: orders
        });
    } catch (error) {
        return res.status(500).json({ error: error.message || "Internal Server Error", details: error.details });
    }
}

export default getAllOrders;