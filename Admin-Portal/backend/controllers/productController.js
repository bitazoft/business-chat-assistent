import productService from "../services/productService.js";

const addProduct = async (req, res) => {
    try {
        const { name, price, stock, description, seller_id } = req.body;
        const product = await productService.createProduct({
            name,
            price: parseFloat(price),
            stock: parseInt(stock),
            description,
            seller_id
          });

          res.status(201).json({
            message: `${product.name} successfully created !!`,
            product: product
          });
    } catch (error) {
        return res.status(500).json({ error: error.message || "Internal Server Error", details: error.details });
    }
}

const deleteProduct = async (req, res) => {
    try {
        const productId = parseInt(req.params.id);
        const product = await productService.deleteProduct(productId);

          res.status(200).json({
            message: `${product.name} successfully deleted !!`,
            product: product
          });
    } catch (error) {
        return res.status(500).json({ error: error.message || "Internal Server Error", details: error.details });
    }
}

const getAllProduct = async (req, res) => {
    try {
        const seller_id = parseInt(req.params.id);
        const products = await productService.getAllProducts(seller_id);

        res.status(200).json({
            message: `All products fetched successfully !!`,
            products: products
        });
    } catch (error) {
        return res.status(500).json({ error: error.message || "Internal Server Error", details: error.details });
    }
}

const getProductById = async (req, res) => {
    try {
        const productId = parseInt(req.params.id);
        const product = await productService.getProductById(productId);

          res.status(200).json({
            message: `${product.name} successfully fetched !!`,
            product: product
          });
    } catch (error) {
        return res.status(500).json({ error: error.message || "Internal Server Error", details: error.details });
    }
}

const updateProduct = async (req, res) => {
    try {
        const { name, price, stock, description, seller_id } = req.body;
        const productId = parseInt(req.params.id);
        const product = await productService.updateProduct(productId,{
            name,
            price: parseFloat(price),
            stock: parseInt(stock),
            description,
            seller_id
          });

          res.status(200).json({
            message: `${product.name} successfully updated !!`,
            product: product
          });
    } catch (error) {
        return res.status(500).json({ error: error.message || "Internal Server Error", details: error.details });
    }
}

export { addProduct, deleteProduct, getAllProduct, getProductById, updateProduct };