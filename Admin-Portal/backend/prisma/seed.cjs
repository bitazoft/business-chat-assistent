const { PrismaClient } = require('@prisma/client');
const prisma = new PrismaClient();

async function main() {
  // Admins
  await prisma.users.createMany({
    data: [
      {
        name: 'Admin One',
        email: 'admin1@bizbot.io',
        phone: '0700000001',
        password: 'admin_pw1',
        role: 'admin',
        address: 'Admin HQ',
      },
      {
        name: 'Admin Two',
        email: 'admin2@bizbot.io',
        phone: '0700000002',
        password: 'admin_pw2',
        role: 'admin',
        address: 'Admin HQ',
      },
    ],
  });

  // Customers
  await prisma.users.createMany({
    data: [
      {
        name: 'Daniel Silva',
        email: 'daniel.silva@email.com',
        phone: '0771234567',
        password: 'cust_pw1',
        role: 'customer',
        address: '22 Rose Garden',
      },
      {
        name: 'Ella Thomas',
        email: 'ella.thomas@email.com',
        phone: '0777654321',
        password: 'cust_pw2',
        role: 'customer',
        address: '17 Lavender St',
      },
    ],
  });

  // Sellers + Profiles + Products
  const sellerData = [
    {
      name: 'Olivia Bennett',
      email: 'olivia.bennett@freshmart.com',
      phone: '0711234567',
      password: 'hashed_pw1',
      address: '123 Maple Street',
      shopName: 'FreshMart Grocery',
      gstNumber: 'GST112233',
      products: [
        { name: 'Basmati Rice 5kg', description: 'Premium long-grain basmati rice', price: 1200, stock: 100 },
        { name: 'Sunflower Oil 1L', description: 'Pure sunflower cooking oil', price: 780, stock: 150 },
        { name: 'Organic Eggs (12 pack)', description: 'Farm fresh eggs', price: 520, stock: 60 },
        { name: 'Whole Wheat Bread', description: 'Baked fresh daily', price: 200, stock: 100 },
        { name: 'Green Lentils 1kg', description: 'Nutritious and organic', price: 350, stock: 90 },
        { name: 'Full Cream Milk 1L', description: 'Dairy milk', price: 240, stock: 130 },
        { name: 'Tomato Ketchup 500ml', description: 'No added preservatives', price: 320, stock: 120 },
        { name: 'Salt 1kg', description: 'Refined iodized salt', price: 90, stock: 300 },
        { name: 'Black Pepper 100g', description: 'Ground pepper spice', price: 400, stock: 70 },
        { name: 'Green Tea Bags', description: 'Pack of 25', price: 250, stock: 200 },
      ],
    },
    {
      name: 'Ethan Ross',
      email: 'ethan.ross@techhub.lk',
      phone: '0712345678',
      password: 'hashed_pw2',
      address: '456 Oak Avenue',
      shopName: 'TechHub Electronics',
      gstNumber: 'GST112234',
      products: [
        { name: 'Wireless Mouse', description: 'Ergonomic design with USB receiver', price: 1850, stock: 50 },
        { name: 'Bluetooth Speaker', description: 'Portable with rich bass', price: 4200, stock: 40 },
        { name: '32GB USB Flash Drive', description: 'Fast transfer speed', price: 1350, stock: 75 },
        { name: 'Gaming Headset', description: 'Noise canceling mic', price: 6500, stock: 30 },
        { name: 'Laptop Cooling Pad', description: 'Silent dual fans', price: 2300, stock: 60 },
        { name: 'HDMI Cable 1.5m', description: 'High-speed 4K support', price: 800, stock: 120 },
        { name: 'Power Bank 10000mAh', description: 'Dual USB output', price: 2900, stock: 35 },
        { name: 'Smartphone Stand', description: 'Adjustable desk holder', price: 950, stock: 80 },
        { name: 'Webcam 1080p', description: 'For video calls', price: 3400, stock: 25 },
        { name: 'Mechanical Keyboard', description: 'RGB backlit, blue switches', price: 8700, stock: 20 },
      ],
    },
    {
      name: 'Ava Mitchell',
      email: 'ava.m@trendystore.com',
      phone: '0713456789',
      password: 'hashed_pw3',
      address: '789 Pine Road',
      shopName: 'Trendy Style Boutique',
      gstNumber: 'GST112235',
      products: [
        { name: 'Floral Maxi Dress', description: 'Elegant summer wear', price: 4500, stock: 25 },
        { name: "Men's Slim Fit Jeans", description: 'Classic dark blue denim', price: 3200, stock: 40 },
        { name: "Women's Blazer", description: 'Formal wear in navy', price: 5400, stock: 15 },
        { name: 'Cotton T-Shirts (Pack of 3)', description: 'Assorted colors', price: 2800, stock: 50 },
        { name: 'Leather Wallet', description: 'Handcrafted brown leather', price: 2200, stock: 35 },
        { name: 'Ankle Boots', description: 'Trendy for all seasons', price: 6900, stock: 20 },
        { name: 'Sunglasses', description: 'UV-protected, black frame', price: 1900, stock: 60 },
        { name: 'Silk Scarf', description: 'Printed with floral patterns', price: 1500, stock: 45 },
        { name: "Men's Casual Shirt", description: 'Checks, full sleeve', price: 2600, stock: 30 },
        { name: "Women's Clutch Bag", description: 'Evening wear', price: 3300, stock: 25 },
      ],
    },
    {
      name: 'Noah Cooper',
      email: 'noah.c@electronics.lk',
      phone: '0714567890',
      password: 'hashed_pw4',
      address: '321 Cedar Blvd',
      shopName: 'ElectroPlus Store',
      gstNumber: 'GST112236',
      products: [
        { name: 'LED Monitor 24"', description: 'Full HD display', price: 18500, stock: 20 },
        { name: 'WiFi Router Dual Band', description: 'Up to 1200 Mbps', price: 7600, stock: 30 },
        { name: 'Wireless Earbuds', description: 'With noise isolation', price: 5800, stock: 40 },
        { name: 'Smartwatch', description: 'Fitness tracking included', price: 8900, stock: 25 },
        { name: 'Power Strip 4-Socket', description: 'Surge protection', price: 2400, stock: 60 },
        { name: 'Laptop Backpack', description: 'Water-resistant', price: 3600, stock: 35 },
        { name: 'USB-C Charging Cable', description: 'Fast charging', price: 850, stock: 100 },
        { name: 'Bluetooth Keyboard', description: 'Ultra-slim design', price: 4200, stock: 30 },
        { name: 'LED Light Strip', description: 'RGB, 5 meters', price: 2800, stock: 45 },
        { name: 'Webcam Cover', description: 'Privacy slider, 3-pack', price: 650, stock: 80 },
      ],
    },
    {
      name: 'Isabella Perez',
      email: 'isa.p@organicstore.com',
      phone: '0715678901',
      password: 'hashed_pw5',
      address: '654 Birch Lane',
      shopName: 'Organic Living',
      gstNumber: 'GST112237',
      products: [
        { name: 'Organic Coconut Oil 500ml', description: 'Cold pressed', price: 950, stock: 60 },
        { name: 'Herbal Green Tea', description: 'Loose leaf, 250g', price: 1250, stock: 70 },
        { name: 'Natural Honey 750ml', description: 'Unfiltered, raw', price: 1650, stock: 40 },
        { name: 'Chia Seeds 500g', description: 'Rich in omega-3', price: 1100, stock: 80 },
        { name: 'Multigrain Flour 1kg', description: 'With 5 grains', price: 1350, stock: 50 },
        { name: 'Almond Butter 300g', description: 'No added sugar', price: 1800, stock: 35 },
        { name: 'Turmeric Powder 250g', description: 'Farm sourced', price: 700, stock: 75 },
        { name: 'Neem Soap', description: 'Antibacterial, handmade', price: 450, stock: 100 },
        { name: 'Amla Juice 1L', description: 'Natural vitamin C booster', price: 1350, stock: 45 },
        { name: 'Organic Face Pack', description: 'With sandalwood', price: 1600, stock: 20 },
      ],
    },
    {
      name: 'Mason Lee',
      email: 'mason.lee@urbanstyle.com',
      phone: '0716789012',
      password: 'hashed_pw6',
      address: '987 Spruce Way',
      shopName: 'UrbanStyle Fashion',
      gstNumber: 'GST112238',
      products: [
        { name: "Men's Leather Jacket", description: 'Genuine leather, black', price: 9800, stock: 15 },
        { name: "Women's Kurta Set", description: 'Cotton fabric', price: 4600, stock: 20 },
        { name: 'Jogger Pants', description: 'Unisex, stretchable', price: 2800, stock: 35 },
        { name: 'Printed Maxi Skirt', description: 'Boho style', price: 3300, stock: 30 },
        { name: 'Denim Jacket', description: 'Unisex, faded wash', price: 5200, stock: 25 },
        { name: "Men's Polo T-Shirts", description: 'Pack of 2', price: 2400, stock: 50 },
        { name: "Ladies Handbag", description: 'Vegan leather', price: 3900, stock: 30 },
        { name: 'Flat Sandals', description: 'For daily wear', price: 2500, stock: 40 },
        { name: 'Sports Shoes', description: 'Running shoes with grip', price: 4600, stock: 20 },
        { name: 'Unisex Beanie Cap', description: 'Wool blend', price: 1200, stock: 60 },
      ],
    },
    {
      name: 'Sophia Hill',
      email: 'sophia.h@kidsworld.lk',
      phone: '0717890123',
      password: 'hashed_pw7',
      address: '147 Elm Drive',
      shopName: 'Kids World Toys',
      gstNumber: 'GST112239',
      products: [
        { name: 'Remote Control Car', description: 'Rechargeable battery included', price: 4200, stock: 40 },
        { name: 'Educational Alphabet Blocks', description: 'Wooden puzzle blocks', price: 1800, stock: 70 },
        { name: 'Baby Rattle Set', description: 'Safe for newborns', price: 950, stock: 100 },
        { name: 'Coloring Kit', description: 'With crayons and coloring book', price: 1100, stock: 80 },
        { name: 'Toy Kitchen Set', description: 'Miniature cooking tools', price: 3200, stock: 30 },
        { name: 'Stuffed Teddy Bear', description: 'Soft and huggable, 18 inches', price: 2700, stock: 45 },
        { name: 'Building Blocks Set', description: 'Compatible with Lego', price: 3400, stock: 25 },
        { name: 'Kids Play Tent', description: 'Foldable and colorful', price: 4900, stock: 15 },
        { name: 'Musical Toy Piano', description: '8 keys with animal sounds', price: 3100, stock: 20 },
        { name: 'Dinosaur Action Figures', description: 'Set of 6', price: 2200, stock: 35 },
      ],
    },
    {
      name: 'Liam Reed',
      email: 'liam.r@homeessentials.com',
      phone: '0718901234',
      password: 'hashed_pw8',
      address: '258 Poplar Street',
      shopName: 'Home Essentials',
      gstNumber: 'GST112240',
      products: [
        { name: 'Non-Stick Frying Pan', description: 'Induction compatible', price: 3500, stock: 30 },
        { name: 'Double Bedsheet Set', description: 'Includes pillowcases', price: 2700, stock: 40 },
        { name: 'Kitchen Knife Set', description: '6 pieces stainless steel', price: 4100, stock: 25 },
        { name: 'Wall Clock', description: 'Silent sweep movement', price: 1800, stock: 50 },
        { name: 'Storage Boxes', description: 'Stackable, set of 3', price: 2500, stock: 60 },
        { name: 'Electric Kettle 1.5L', description: 'Auto shut-off feature', price: 3200, stock: 35 },
        { name: 'Bathroom Mat Set', description: 'Anti-slip, 2 pieces', price: 1700, stock: 40 },
        { name: 'LED Desk Lamp', description: 'Touch control with USB', price: 2900, stock: 30 },
        { name: 'Laundry Basket', description: 'Foldable with lid', price: 2200, stock: 20 },
        { name: 'Curtain Rod Set', description: 'Adjustable size', price: 1900, stock: 25 },
      ],
    },
    {
      name: 'Mia Turner',
      email: 'mia.turner@beauty.lk',
      phone: '0719012345',
      password: 'hashed_pw9',
      address: '369 Ash Terrace',
      shopName: 'Beauty & Beyond',
      gstNumber: 'GST112241',
      products: [
        { name: 'Aloe Vera Gel 250ml', description: 'Soothing and moisturizing', price: 850, stock: 100 },
        { name: 'Matte Lipstick Set', description: 'Pack of 3', price: 2900, stock: 70 },
        { name: 'Vitamin C Serum', description: 'Brightens skin', price: 3200, stock: 50 },
        { name: 'Facial Cleanser', description: 'Gentle daily use', price: 1800, stock: 90 },
        { name: 'Compact Powder', description: 'Lightweight, long-lasting', price: 2100, stock: 60 },
        { name: 'Nail Polish Set', description: '5 assorted colors', price: 2400, stock: 45 },
        { name: 'Hair Conditioner', description: 'Hydrating formula', price: 1700, stock: 80 },
        { name: 'Perfume 50ml', description: 'Floral scent', price: 4700, stock: 30 },
        { name: 'Makeup Brushes Set', description: '10 pieces synthetic', price: 3600, stock: 40 },
        { name: 'Eye Shadow Palette', description: '12 shades', price: 3900, stock: 25 },
      ],
    },
    {
      name: 'James King',
      email: 'james.king@fitness.lk',
      phone: '0710123456',
      password: 'hashed_pw10',
      address: '753 Willow Crescent',
      shopName: 'FitLife Gear',
      gstNumber: 'GST112242',
      products: [
        { name: 'Yoga Mat', description: 'Non-slip surface', price: 3200, stock: 50 },
        { name: 'Dumbbell Set 10kg', description: 'Adjustable weights', price: 8700, stock: 30 },
        { name: 'Protein Powder 2kg', description: 'Whey isolate', price: 12500, stock: 40 },
        { name: 'Fitness Tracker', description: 'Heart rate monitor', price: 9800, stock: 35 },
        { name: 'Running Shoes', description: 'Lightweight with cushioning', price: 7800, stock: 25 },
        { name: 'Resistance Bands', description: 'Set of 5 varying tension', price: 2200, stock: 60 },
        { name: 'Water Bottle 1L', description: 'BPA free', price: 900, stock: 80 },
        { name: 'Jump Rope', description: 'Adjustable length', price: 1500, stock: 70 },
        { name: 'Gym Gloves', description: 'Breathable material', price: 1800, stock: 45 },
        { name: 'Foam Roller', description: 'Muscle recovery aid', price: 3200, stock: 20 },
      ],
    },
  ];

  for (const seller of sellerData) {
    const user = await prisma.users.create({
      data: {
        name: seller.name,
        email: seller.email,
        phone: seller.phone,
        password: seller.password,
        role: 'seller',
        address: seller.address,
      },
    });

    await prisma.sellerProfile.create({
      data: {
        userId: user.id,
        shopName: seller.shopName,
        gstNumber: seller.gstNumber,
      },
    });

    for (const product of seller.products) {
      await prisma.products.create({
        data: {
          sellerId: user.id,
          name: product.name,
          description: product.description,
          price: product.price,
          stock: product.stock,
        },
      });
    }
  }

  console.log('Seeding completed!');
}

main()
  .catch((e) => {
    console.error(e);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });
