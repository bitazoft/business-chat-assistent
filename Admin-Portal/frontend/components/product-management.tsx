"use client"

import { useEffect, useState } from "react"
import { Plus, Edit, Trash2, Search, Package } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { AddProductModal } from "@/components/add-product-modal"
import { EditProductModal } from "@/components/edit-product-modal"
import { getCurrentUser } from "@/lib/auth"
import { toast } from "sonner"

export interface Product {
  id: number
  name: string
  price: string
  stock: number
  category: string
  status: string
  description?: string
}

export function ProductManagement() {
  const [products, setProducts] = useState<Product[]>([])
  const [searchTerm, setSearchTerm] = useState("")
  const [isAddModalOpen, setIsAddModalOpen] = useState(false)
  const [isEditModalOpen, setIsEditModalOpen] = useState(false)
  const [selectedProduct, setSelectedProduct] = useState<Product | null>(null)
  const user = getCurrentUser()

  useEffect(() => {
    fetchProducts()
  }, []);

  const fetchProducts = async () => {
    fetch(`http://localhost:7001/api/products/getAll/${user?.id}`,{
      method: "GET",
      credentials: "include",
      headers: {
        "Content-Type": "application/json",
      },
    })
      .then((res) => res.json())
      .then((data) => {
        console.log("Fetched products", data);
        setProducts(data.products || []);
      })
      .catch((err) => {
        toast.error("Failed to fetch products", err);
      });
  }

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

  const handleDeleteProduct = async (productId: number) => {
    try {
      await fetch(`http://localhost:7001/api/products/delete/${productId}`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
      });
      
      fetchProducts()

      toast.success("Product deleted successfully!",{
        style: {
          background: "#0f0f23",
          color: "#fff",
        }
      });
    } catch (error) {
      toast.error((error as Error).toString());
    }
  }

  // Empty function for add product - you can implement your logic here
  const onAddProduct = async (productData: {
    name: string
    price: string
    description: string
    stock: number
  }) => {
    try {
      await fetch(`http://localhost:7001/api/products/add`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: productData.name,
          price: productData.price,
          description: productData.description,
          stock: productData.stock,
          seller_id: user?.id
        }),
      });

      fetchProducts()
      setIsAddModalOpen(false)

      toast.success("Product saved successfully !!",{
        style: {
          background: "#0f0f23",
          color: "#fff",
        }
      });
    } catch (error) {
      toast.error((error as Error).toString());
    }
  }

  // Empty function for update product - you can implement your logic here
  const onUpdateProduct = async (productData: {
    name: string
    price: string
    description: string
    stock: number
  }) => {
    try {
      await fetch(`http://localhost:7001/api/products/update/${selectedProduct?.id}`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: productData.name,
          price: productData.price,
          description: productData.description,
          stock: productData.stock,
          seller_id: user?.id
        }),
      });

      fetchProducts()
      setIsEditModalOpen(false)
      setSelectedProduct(null)

      toast.success("Product details updated successfully!",{
        style: {
          background: "#0f0f23",
          color: "#fff",
        }
      });
    } catch (error) {
      toast.error((error as Error).toString());
    }
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
          {filteredProducts.length === 0 ? (
            // No products message
            <div className="flex flex-col items-center justify-center py-12 text-center">
              <Package className="w-16 h-16 text-gray-600 mb-4" />
              <h3 className="text-xl font-semibold text-gray-400 mb-2">
                {searchTerm ? "No products found" : "No products available"}
              </h3>
              <p className="text-gray-500 mb-6 max-w-md">
                {searchTerm
                  ? `No products match your search for "${searchTerm}". Try adjusting your search terms.`
                  : "You haven't added any products yet. Start by adding your first product to get started."}
              </p>
              {!searchTerm && (
                <Button
                  onClick={handleAddProduct}
                  className="bg-gradient-to-r from-violet-500 to-purple-600 hover:from-violet-600 hover:to-purple-700 text-white shadow-lg hover:shadow-violet-500/25 transition-all duration-300"
                >
                  <Plus className="w-4 h-4 mr-2" />
                  Add Your First Product
                </Button>
              )}
            </div>
          ) : (
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
                    <td className="py-3 px-4 text-gray-300">{product.category || "General"}</td>
                    <td className="py-3 px-4">
                      <span className={`px-2 py-1 rounded-full text-xs ${getStatusColor(product.stock > 20 ? "Active":product.stock < 20 && product.stock != 0 ? "Low Stock":"Out of Stock" )}`}>
                        {product.stock > 20 ? "Active":product.stock < 20 && product.stock != 0 ? "Low Stock":"Out of Stock" }
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
          )}
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
