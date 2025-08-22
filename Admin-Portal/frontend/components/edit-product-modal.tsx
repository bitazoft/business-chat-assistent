"use client"

import type React from "react"

import { useState, useEffect } from "react"
import { createPortal } from "react-dom"
import { X, Edit } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import type { Product } from "@/components/product-management"

interface EditProductModalProps {
  isOpen: boolean
  onClose: () => void
  onUpdateProduct: (productData: {
    name: string
    price: string
    description: string
    stock: number
  }) => void
  product: Product | null
}

export function EditProductModal({ isOpen, onClose, onUpdateProduct, product }: EditProductModalProps) {
  const [mounted, setMounted] = useState(false)
  const [formData, setFormData] = useState({
    name: "",
    price: "",
    description: "",
    stock: 0,
  })

  const [errors, setErrors] = useState({
    name: "",
    price: "",
    description: "",
    stock: "",
  })

  // Ensure component is mounted before rendering portal
  useEffect(() => {
    setMounted(true)
  }, [])

  // Set form data when modal opens with product data
  useEffect(() => {
    if (isOpen && product) {
      setFormData({
        name: product.name,
        price: product.price,
        description: product.description || "",
        stock: product.stock,
      })
      setErrors({
        name: "",
        price: "",
        description: "",
        stock: "",
      })
    }
  }, [isOpen, product])

  // Blur effect - ONLY blur main content and header
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = "hidden"

      // Target specific elements to blur
      const mainContent = document.querySelector("main")
      const topbar = document.querySelector("header")

      // ONLY blur main content and topbar
      if (mainContent) mainContent.classList.add("blur-sm")
      if (topbar) topbar.classList.add("blur-sm")
    } else {
      document.body.style.overflow = "unset"

      const mainContent = document.querySelector("main")
      const topbar = document.querySelector("header")

      if (mainContent) mainContent.classList.remove("blur-sm")
      if (topbar) topbar.classList.remove("blur-sm")
    }

    // Cleanup on unmount
    return () => {
      document.body.style.overflow = "unset"

      const mainContent = document.querySelector("main")
      const topbar = document.querySelector("header")

      if (mainContent) mainContent.classList.remove("blur-sm")
      if (topbar) topbar.classList.remove("blur-sm")
    }
  }, [isOpen])

  const validateForm = () => {
    const newErrors = {
      name: "",
      price: "",
      description: "",
      stock: "",
    }

    if (!formData.name.trim()) {
      newErrors.name = "Product name is required"
    }

    if (!formData.price) {
      newErrors.price = "Price is required"
    }

    if (parseFloat(formData.price) < 0) {
      newErrors.price = "Price cannot be negative"
    }

    if (!formData.description.trim()) {
      newErrors.description = "Description is required"
    }

    if (formData.stock < 0) {
      newErrors.stock = "Stock cannot be negative"
    }

    setErrors(newErrors)
    return !Object.values(newErrors).some((error) => error !== "")
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()

    if (validateForm()) {
      onUpdateProduct(formData)
    }
  }

  const handleInputChange = (field: string, value: string | number) => {
    setFormData((prev) => ({
      ...prev,
      [field]: value,
    }))

    if (errors[field as keyof typeof errors]) {
      setErrors((prev) => ({
        ...prev,
        [field]: "",
      }))
    }
  }

  const handleBackdropClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) {
      onClose()
    }
  }

  // Don't render anything if not mounted or not open
  if (!mounted || !isOpen || !product) return null

  // Modal content
  const modalContent = (
    <div
      className="fixed inset-0 bg-black/60 flex items-center justify-center z-[9999] p-4 animate-in fade-in-0 duration-300"
      onClick={handleBackdropClick}
    >
      <div className="bg-gradient-to-b from-[#1a1a2e] to-[#16213e] rounded-xl border border-gray-700 w-full max-w-md shadow-2xl animate-in zoom-in-95 duration-300">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-700">
          <h2 className="text-xl font-bold text-white flex items-center">
            <Edit className="w-5 h-5 mr-2 text-violet-400" />
            Edit Product
          </h2>
          <Button
            variant="ghost"
            size="icon"
            onClick={onClose}
            className="text-gray-400 hover:text-white hover:bg-gray-800/50 transition-colors duration-200"
          >
            <X className="w-5 h-5" />
          </Button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="p-6 space-y-4">
          {/* Product Name */}
          <div className="space-y-2">
            <Label htmlFor="editProductName" className="text-violet-400 font-medium">
              Product Name *
            </Label>
            <Input
              id="editProductName"
              type="text"
              value={formData.name}
              onChange={(e) => handleInputChange("name", e.target.value)}
              className={`bg-[#0f0f23] border-gray-600 text-white focus:border-violet-400 focus:ring-violet-400/20 transition-all duration-300 ${
                errors.name ? "border-red-500" : ""
              }`}
              placeholder="Enter product name"
              autoFocus
            />
            {errors.name && <p className="text-red-400 text-sm">{errors.name}</p>}
          </div>

          {/* Price */}
          <div className="space-y-2">
            <Label htmlFor="editProductPrice" className="text-violet-400 font-medium">
              Price *
            </Label>
            <Input
              id="editProductPrice"
              type="text"
              value={formData.price}
              onChange={(e) => handleInputChange("price", e.target.value)}
              className={`bg-[#0f0f23] border-gray-600 text-white focus:border-violet-400 focus:ring-violet-400/20 transition-all duration-300 ${
                errors.price ? "border-red-500" : ""
              }`}
              placeholder="e.g., $99.99"
            />
            {errors.price && <p className="text-red-400 text-sm">{errors.price}</p>}
          </div>

          {/* Stock */}
          <div className="space-y-2">
            <Label htmlFor="editProductStock" className="text-violet-400 font-medium">
              Stock Quantity *
            </Label>
            <Input
              id="editProductStock"
              type="number"
              min="0"
              value={formData.stock}
              onChange={(e) => handleInputChange("stock", Number.parseInt(e.target.value) || 0)}
              className={`bg-[#0f0f23] border-gray-600 text-white focus:border-violet-400 focus:ring-violet-400/20 transition-all duration-300 ${
                errors.stock ? "border-red-500" : ""
              }`}
              placeholder="Enter stock quantity"
            />
            {errors.stock && <p className="text-red-400 text-sm">{errors.stock}</p>}
          </div>

          {/* Description */}
          <div className="space-y-2">
            <Label htmlFor="editProductDescription" className="text-violet-400 font-medium">
              Description *
            </Label>
            <Textarea
              id="editProductDescription"
              value={formData.description}
              onChange={(e) => handleInputChange("description", e.target.value)}
              className={`bg-[#0f0f23] border-gray-600 text-white focus:border-violet-400 focus:ring-violet-400/20 transition-all duration-300 min-h-[100px] resize-none ${
                errors.description ? "border-red-500" : ""
              }`}
              placeholder="Enter product description"
            />
            {errors.description && <p className="text-red-400 text-sm">{errors.description}</p>}
          </div>

          {/* Buttons */}
          <div className="flex space-x-3 pt-4">
            <Button
              type="button"
              variant="outline"
              onClick={onClose}
              className="flex-1 border-gray-600 text-gray-300 hover:text-white hover:border-gray-500 bg-transparent transition-colors duration-200"
            >
              Cancel
            </Button>
            <Button
              type="submit"
              className="flex-1 bg-gradient-to-r from-violet-500 to-purple-600 hover:from-violet-600 hover:to-purple-700 text-white shadow-lg hover:shadow-violet-500/25 transition-all duration-300"
            >
              Update Product
            </Button>
          </div>
        </form>
      </div>
    </div>
  )

  // Render modal using portal to document.body
  return createPortal(modalContent, document.body)
}
