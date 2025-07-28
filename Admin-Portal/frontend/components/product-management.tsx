"use client"

import { useState } from "react"
import { Plus, Edit, Trash2, Search } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { AddProductModal } from "@/components/add-product-modal"
import { EditProductModal } from "@/components/edit-product-modal"

export interface Product {
  id: number
  name: string
  price: string
  stock: number
  category: string
  status: string
  description?: string
}

const initialProducts: Product[] = [
  {
    id: 1,
    name: "iPhone 15 Pro",
    price: "$999",
    stock: 25,
    category: "Electronics",
    status: "Active",
    description: "Latest iPhone with advanced camera system and titanium design",
  },
  {
    id: 2,
    name: "Samsung Galaxy S24",
    price: "$799",
    stock: 18,
    category: "Electronics",
    status: "Active",
    description: "Flagship Android phone with AI-powered features",
  },
  {
    id: 3,
    name: "MacBook Air M3",
    price: "$1,199",
    stock: 12,
    category: "Computers",
    status: "Active",
    description: "Ultra-thin laptop with M3 chip for exceptional performance",
  },
  {
    id: 4,
    name: "AirPods Pro",
    price: "$249",
    stock: 45,
    category: "Accessories",
    status: "Low Stock",
    description: "Wireless earbuds with active noise cancellation",
  },
  {
    id: 5,
    name: "iPad Air",
    price: "$599",
    stock: 0,
    category: "Tablets",
    status: "Out of Stock",
    description: "Versatile tablet perfect for work and creativity",
  },
]

export function ProductManagement() {
  const [products, setProducts] = useState<Product[]>(initialProducts)
  const [searchTerm, setSearchTerm] = useState("")
  const [isAddModalOpen, setIsAddModalOpen] = useState(false)
  const [isEditModalOpen, setIsEditModalOpen] = useState(false)
  const [selectedProduct, setSelectedProduct] = useState<Product | null>(null)

  const filteredProducts = products.filter((product) => product.name.toLowerCase().includes(searchTerm.toLowerCase()))

  const getStatusColor = (status: string) => {
    switch (status) {
      case "Active":
        return "bg-emerald-500/20 text-emerald-400"
      case "Low Stock":
        return "bg-yellow-500/20 text-yellow-400"
      case "Out of Stock":
        return "bg-red-500/20 text-red-400"
      default:
        return "bg-gray-500/20 text-gray-400"
    }
  }

  const handleAddProduct = () => {
    setIsAddModalOpen(true)
  }

  const handleEditProduct = (product: Product) => {
    setSelectedProduct(product)
    setIsEditModalOpen(true)
  }

  const handleDeleteProduct = (productId: number) => {
    // TODO: Implement delete logic here
    console.log("Delete product with ID:", productId)

    // For now, just remove from local state
    setProducts(products.filter((product) => product.id !== productId))
  }

  // Empty function for add product - you can implement your logic here
  const onAddProduct = (productData: {
    name: string
    price: string
    description: string
    stock: number
  }) => {
    // TODO: Implement your add product logic here
    // This could include API calls, validation, etc.
    console.log("Add product data:", productData)

    // For demo purposes, add to local state
    const newProduct: Product = {
      id: Math.max(...products.map((p) => p.id)) + 1,
      name: productData.name,
      price: productData.price,
      stock: productData.stock,
      category: "General", // You can add category selection later
      status: productData.stock > 0 ? "Active" : "Out of Stock",
      description: productData.description,
    }

    setProducts([...products, newProduct])
    setIsAddModalOpen(false)
  }

  // Empty function for update product - you can implement your logic here
  const onUpdateProduct = (productData: {
    name: string
    price: string
    description: string
    stock: number
  }) => {
    // TODO: Implement your update product logic here
    // This could include API calls, validation, etc.
    console.log("Update product data:", productData)
    console.log("Selected product ID:", selectedProduct?.id)

    // For demo purposes, update local state
    if (selectedProduct) {
      const updatedProducts = products.map((product) =>
        product.id === selectedProduct.id
          ? {
              ...product,
              name: productData.name,
              price: productData.price,
              stock: productData.stock,
              description: productData.description,
              status: productData.stock > 0 ? "Active" : "Out of Stock",
            }
          : product,
      )
      setProducts(updatedProducts)
    }

    setIsEditModalOpen(false)
    setSelectedProduct(null)
  }

  return (
    <div className="space-y-6 animate-in fade-in-50 duration-500">
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-3xl font-bold text-white mb-2">Product Management</h2>
          <p className="text-gray-400">Manage your product catalog</p>
        </div>
        <Button
          onClick={handleAddProduct}
          className="bg-gradient-to-r from-violet-500 to-purple-600 hover:from-violet-600 hover:to-purple-700 text-white shadow-lg hover:shadow-violet-500/25 transition-all duration-300"
        >
          <Plus className="w-4 h-4 mr-2" />
          Add Product
        </Button>
      </div>

      {/* Search */}
      <div className="relative animate-in slide-in-from-top duration-500">
        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
        <Input
          placeholder="Search products..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="pl-10 bg-[#1a1a2e] border-gray-600 text-white focus:border-violet-400 focus:ring-violet-400/20 transition-all duration-300"
        />
      </div>

      {/* Products Table */}
      <Card className="bg-gradient-to-br from-[#1a1a2e] to-[#16213e] border-gray-700 animate-in slide-in-from-bottom duration-700">
        <CardHeader>
          <CardTitle className="text-white">Products ({filteredProducts.length})</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-700">
                  <th className="text-left py-3 px-4 text-violet-400 font-semibold">Product Name</th>
                  <th className="text-left py-3 px-4 text-violet-400 font-semibold">Price</th>
                  <th className="text-left py-3 px-4 text-violet-400 font-semibold">Stock</th>
                  <th className="text-left py-3 px-4 text-violet-400 font-semibold">Category</th>
                  <th className="text-left py-3 px-4 text-violet-400 font-semibold">Status</th>
                  <th className="text-left py-3 px-4 text-violet-400 font-semibold">Actions</th>
                </tr>
              </thead>
              <tbody>
                {filteredProducts.map((product, index) => (
                  <tr
                    key={product.id}
                    className="border-b border-gray-800 hover:bg-gray-800/30 transition-all duration-200 animate-in slide-in-from-left"
                    style={{ animationDelay: `${index * 50}ms` }}
                  >
                    <td className="py-3 px-4 text-white font-medium">{product.name}</td>
                    <td className="py-3 px-4 text-emerald-400 font-semibold">{product.price}</td>
                    <td className="py-3 px-4 text-gray-300">{product.stock}</td>
                    <td className="py-3 px-4 text-gray-300">{product.category}</td>
                    <td className="py-3 px-4">
                      <span className={`px-2 py-1 rounded-full text-xs ${getStatusColor(product.status)}`}>
                        {product.status}
                      </span>
                    </td>
                    <td className="py-3 px-4">
                      <div className="flex space-x-2">
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => handleEditProduct(product)}
                          className="text-violet-400 hover:text-emerald-400 hover:bg-gray-800/50"
                        >
                          <Edit className="w-4 h-4" />
                        </Button>
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => handleDeleteProduct(product.id)}
                          className="text-red-400 hover:text-red-300 hover:bg-gray-800/50"
                        >
                          <Trash2 className="w-4 h-4" />
                        </Button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {/* Add Product Modal */}
      <AddProductModal isOpen={isAddModalOpen} onClose={() => setIsAddModalOpen(false)} onAddProduct={onAddProduct} />

      {/* Edit Product Modal */}
      <EditProductModal
        isOpen={isEditModalOpen}
        onClose={() => {
          setIsEditModalOpen(false)
          setSelectedProduct(null)
        }}
        onUpdateProduct={onUpdateProduct}
        product={selectedProduct}
      />
    </div>
  )
}
