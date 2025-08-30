"use client";

import { useEffect, useState } from "react";
import {
  Plus,
  Edit,
  Trash2,
  Search,
  Package,
  ChevronRight,
  ChevronLeft,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { AddProductModal } from "@/components/add-product-modal";
import { EditProductModal } from "@/components/edit-product-modal";
import { getCurrentUser } from "@/lib/auth";
import { toast } from "sonner";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { json } from "stream/consumers";

export interface Product {
  id: number;
  name: string;
  price: string;
  stock: number;
  category: string;
  status: string;
  description?: string;
  image_url: string;
}

export function ProductManagement() {
  const [products, setProducts] = useState<Product[]>([]);
  const [searchTerm, setSearchTerm] = useState("");
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage, setItemsPerPage] = useState(5);
  const [isAddModalOpen, setIsAddModalOpen] = useState(false);
  const [isEditModalOpen, setIsEditModalOpen] = useState(false);
  const [selectedProduct, setSelectedProduct] = useState<Product | null>(null);
  const user = getCurrentUser();

  useEffect(() => {
    setCurrentPage(1);
    fetchProducts();
  }, [searchTerm]);

  const fetchProducts = async () => {
    fetch(`http://localhost:7001/api/products/getAll/${user?.sellerId}`, {
      method: "GET",
      credentials: "include",
      headers: {
        "Content-Type": "application/json",
      },
    })
      .then((res) => res.json())
      .then((data) => {
        setProducts(data.products || []);
      })
      .catch((err) => {
        toast.error("Failed to fetch products", err);
      });
  };

  const filteredProducts = products.filter((product) =>
    product.name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  // Pagination calculations
  const totalPages = Math.ceil(filteredProducts.length / itemsPerPage);
  const startIndex = (currentPage - 1) * itemsPerPage;
  const endIndex = startIndex + itemsPerPage;
  const paginatedProducts = filteredProducts.slice(startIndex, endIndex);

  const getStatusColor = (status: string) => {
    switch (status) {
      case "Active":
        return "bg-emerald-500/20 text-emerald-400";
      case "Low Stock":
        return "bg-yellow-500/20 text-yellow-400";
      case "Out of Stock":
        return "bg-red-500/20 text-red-400";
      default:
        return "bg-gray-500/20 text-gray-400";
    }
  };

  const handlePageChange = (page: number) => {
    setCurrentPage(page);
  };

  const handleItemsPerPageChange = (value: string) => {
    setItemsPerPage(Number(value));
    setCurrentPage(1);
  };

  const handleAddProduct = () => {
    setIsAddModalOpen(true);
  };

  const handleEditProduct = (product: Product) => {
    setSelectedProduct(product);
    setIsEditModalOpen(true);
  };

  const handleDeleteProduct = async (productId: number) => {
    try {
      await fetch(`http://localhost:7001/api/products/delete/${productId}`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
      });

      fetchProducts();

      toast.success("Product deleted successfully!", {
        style: {
          background: "#0f0f23",
          color: "#fff",
        },
      });
    } catch (error) {
      toast.error((error as Error).toString());
    }
  };

  const uploadImage = async (file: File) => {
    try {
      const res = await fetch(`http://localhost:7001/api/uploads/image`,{
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          fileName: file.name,
          fileType: file.type,
          fileSize: file.size,
          folder: "products"        
        })
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.error || "Failed to get upload URL");
      }
  
      const { uploadUrl, fileUrl } = await res.json();
      
      const resUpload = await fetch(uploadUrl, {
        method: "PUT",
        headers: {
          "Content-Type": file.type,
        },
        body: file,
      });

      if (!resUpload.ok) {
        const err = await resUpload.json();
        throw new Error(err.error || "Failed to upload image");
      }

      return fileUrl;
    } catch (err) {
      throw err;
    }
  }

  // Empty function for add product - you can implement your logic here
  const onAddProduct = async (productData: {
    name: string;
    price: string;
    description: string;
    stock: number;
    file: File | null
  }) => {
    try {
      const imageUrl = await uploadImage(productData.file!);

      await fetch(`http://localhost:7001/api/products/add`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: productData.name,
          price: productData.price,
          description: productData.description,
          stock: productData.stock,
          image_url: imageUrl,
          seller_id: user?.sellerId,
        }),
      });

      fetchProducts();
      setIsAddModalOpen(false);

      toast.success("Product added successfully !!", {
        style: {
          background: "rgba(0, 128, 0, 0.3)",
          color: "#fff",
        },
      });
    } catch (error) {
      toast.error((error as Error).toString());
    }
  };

  // Empty function for update product - you can implement your logic here
  const onUpdateProduct = async (productData: {
    name: string;
    price: string;
    description: string;
    stock: number;
    file: File | null
    existingImageUrl?: string
  }) => {
    try {
      let imageUrl;
      if (productData.file != null) {
        imageUrl = await uploadImage(productData.file)
      }

      await fetch(
        `http://localhost:7001/api/products/update/${selectedProduct?.id}`,
        {
          method: "POST",
          credentials: "include",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            name: productData.name,
            price: productData.price,
            description: productData.description,
            stock: productData.stock,
            image_url: imageUrl ? imageUrl : productData.existingImageUrl,
            seller_id: user?.sellerId,
          }),
        }
      );

      fetchProducts();
      setIsEditModalOpen(false);
      setSelectedProduct(null);

      toast.success("Product details updated successfully!", {
        style: {
          background: "rgba(0, 128, 0, 0.3)",
          color: "#fff",
        },
      });
    } catch (error) {
      toast.error((error as Error).toString());
    }
  };

  return (
    <div className="space-y-6 animate-in fade-in-50 duration-500">
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-3xl font-bold text-white mb-2">
            Product Management
          </h2>
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
          <CardTitle className="text-white flex items-center justify-between">
            <span>Products ({filteredProducts.length})</span>
            <div className="flex items-center space-x-2 text-sm text-gray-400">
              <span>Show</span>
              <Select
                value={itemsPerPage.toString()}
                onValueChange={handleItemsPerPageChange}
              >
                <SelectTrigger className="w-20 h-8 bg-[#0f0f23] border-gray-600 text-white">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="bg-[#1a1a2e] border-gray-600">
                  <SelectItem
                    value="5"
                    className="text-white hover:bg-gray-800"
                  >
                    5
                  </SelectItem>
                  <SelectItem
                    value="10"
                    className="text-white hover:bg-gray-800"
                  >
                    10
                  </SelectItem>
                  <SelectItem
                    value="25"
                    className="text-white hover:bg-gray-800"
                  >
                    25
                  </SelectItem>
                  <SelectItem
                    value="50"
                    className="text-white hover:bg-gray-800"
                  >
                    50
                  </SelectItem>
                </SelectContent>
              </Select>
              <span>per page</span>
            </div>
          </CardTitle>
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
            <>
              {/* Table */}
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-gray-700">
                      <th className="text-left py-3 px-4 text-violet-400 font-semibold">Image</th>
                      <th className="text-left py-3 px-4 text-violet-400 font-semibold">
                        Product Name
                      </th>
                      <th className="text-left py-3 px-4 text-violet-400 font-semibold">
                        Price
                      </th>
                      <th className="text-left py-3 px-4 text-violet-400 font-semibold">
                        Stock
                      </th>
                      <th className="text-left py-3 px-4 text-violet-400 font-semibold">
                        Category
                      </th>
                      <th className="text-left py-3 px-4 text-violet-400 font-semibold">
                        Status
                      </th>
                      <th className="text-left py-3 px-4 text-violet-400 font-semibold">
                        Actions
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {paginatedProducts.map((product, index) => (
                      <tr
                        key={product.id}
                        className="border-b border-gray-800 hover:bg-gray-800/30 transition-all duration-200 animate-in slide-in-from-left"
                        style={{ animationDelay: `${index * 50}ms` }}
                      >
                      <td className="py-3 px-4">
                        <div className="w-12 h-12 rounded-lg overflow-hidden bg-gray-800 flex items-center justify-center">
                          {product.image_url ? (
                            <img
                              src={product.image_url || "/placeholder.svg"}
                              alt={product.name}
                              className="w-full h-full object-cover"
                              crossOrigin="anonymous"
                            />
                          ) : (
                            <Package className="w-6 h-6 text-gray-500" />
                          )}
                        </div>
                      </td>
                        <td className="py-3 px-4 text-white font-medium">
                          {product.name}
                        </td>
                        <td className="py-3 px-4 text-emerald-400 font-semibold">
                          {product.price}
                        </td>
                        <td className="py-3 px-4 text-gray-300">
                          {product.stock}
                        </td>
                        <td className="py-3 px-4 text-gray-300">
                          {product.category || "General"}
                        </td>
                        <td className="py-3 px-4">
                          <span
                            className={`px-2 py-1 rounded-full text-xs ${getStatusColor(
                              product.stock > 20
                                ? "Active"
                                : product.stock < 20 && product.stock != 0
                                ? "Low Stock"
                                : "Out of Stock"
                            )}`}
                          >
                            {product.stock > 20
                              ? "Active"
                              : product.stock < 20 && product.stock != 0
                              ? "Low Stock"
                              : "Out of Stock"}
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

              {/* Pagination Controls */}
              {totalPages > 1 && (
                <div className="flex items-center justify-between mt-6 pt-4 border-t border-gray-700">
                  <div className="text-sm text-gray-400">
                    Showing {startIndex + 1} to{" "}
                    {Math.min(endIndex, filteredProducts.length)} of{" "}
                    {filteredProducts.length} products
                  </div>

                  <div className="flex items-center space-x-2">
                    {/* Previous Button */}
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handlePageChange(currentPage - 1)}
                      disabled={currentPage === 1}
                      className="border-gray-600 text-gray-300 hover:text-white hover:border-gray-500 bg-transparent disabled:opacity-50"
                    >
                      <ChevronLeft className="w-4 h-4 mr-1" />
                      Previous
                    </Button>

                    {/* Page Numbers */}
                    <div className="flex items-center space-x-1">
                      {Array.from({ length: totalPages }, (_, i) => i + 1).map(
                        (page) => {
                          // Show first page, last page, current page, and pages around current page
                          const showPage =
                            page === 1 ||
                            page === totalPages ||
                            (page >= currentPage - 1 &&
                              page <= currentPage + 1);

                          if (!showPage) {
                            // Show ellipsis for gaps
                            if (
                              page === currentPage - 2 ||
                              page === currentPage + 2
                            ) {
                              return (
                                <span key={page} className="px-2 text-gray-500">
                                  ...
                                </span>
                              );
                            }
                            return null;
                          }

                          return (
                            <Button
                              key={page}
                              variant={
                                currentPage === page ? "default" : "outline"
                              }
                              size="sm"
                              onClick={() => handlePageChange(page)}
                              className={
                                currentPage === page
                                  ? "bg-gradient-to-r from-violet-500 to-purple-600 text-white"
                                  : "border-gray-600 text-gray-300 hover:text-white hover:border-gray-500 bg-transparent"
                              }
                            >
                              {page}
                            </Button>
                          );
                        }
                      )}
                    </div>

                    {/* Next Button */}
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handlePageChange(currentPage + 1)}
                      disabled={currentPage === totalPages}
                      className="border-gray-600 text-gray-300 hover:text-white hover:border-gray-500 bg-transparent disabled:opacity-50"
                    >
                      Next
                      <ChevronRight className="w-4 h-4 ml-1" />
                    </Button>
                  </div>
                </div>
              )}
            </>
          )}
        </CardContent>
      </Card>

      {/* Add Product Modal */}
      <AddProductModal
        isOpen={isAddModalOpen}
        onClose={() => setIsAddModalOpen(false)}
        onAddProduct={onAddProduct}
      />

      {/* Edit Product Modal */}
      <EditProductModal
        isOpen={isEditModalOpen}
        onClose={() => {
          setIsEditModalOpen(false);
          setSelectedProduct(null);
        }}
        onUpdateProduct={onUpdateProduct}
        product={selectedProduct}
      />
    </div>
  );
}
