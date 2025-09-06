export default class ProductResponse {
  constructor({
    id,
    name,
    price,
    stock,
    category,
    status,
    description,
    images,
  }) {
    this.id = id;
    this.name = name;
    this.price = price;
    this.stock = stock;
    this.category = category;
    this.status = status;
    this.description = description || undefined;
    this.images = images?.map((img) => new ImageResponse(img)) || [];
  }
}

class ImageResponse {
  constructor({ id, url, isMain }) {
    this.id = id;
    this.url = url;
    this.isMain = isMain;
  }
}


