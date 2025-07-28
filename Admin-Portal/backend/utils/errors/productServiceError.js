class ProductUpdateError extends Error {
  constructor(message, details) {
    super(message);
    this.name = "ProductServiceError";
    this.details = details;
  }
}

export default ProductUpdateError;
