"use client"

import type React from "react"

import { useState, useEffect, useRef } from "react"
import { createPortal } from "react-dom"
import { X, Edit, Upload, ImageIcon, Trash2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import type { Product } from "@/components/product-management"
import { toast } from "sonner"

interface EditProductModalProps {
  isOpen: boolean
  onClose: () => void
  onUpdateProduct: (productData: {
    name: string
    price: string
    description: string
    stock: number
    file: File | null
    existingImageUrl?: string
  }) => Promise<void>
  product: Product | null
}

export function EditProductModal({ isOpen, onClose, onUpdateProduct, product }: EditProductModalProps) {
  const [mounted, setMounted] = useState(false)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [isDragOver, setIsDragOver] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [formData, setFormData] = useState({
    name: "",
    price: "",
    description: "",
    stock: 0,
    file: null as File | null,
  })

  const [previewUrl, setPreviewUrl] = useState("")
  const [errors, setErrors] = useState({
    name: "",
    price: "",
    description: "",
    stock: "",
    imageUrl: "",
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
        file: null, // Always start with no new file
      })
      setPreviewUrl(product.image_url || "") 
      setErrors({
        name: "",
        price: "",
        description: "",
        stock: "",
        imageUrl: "",
      })
      setIsSubmitting(false)
      setIsDragOver(false)
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
      imageUrl: "",
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

  const handleFileSelect = (file: File) => {
    // Validate file type
    if (!file.type.startsWith("image/")) {
      setErrors((prev) => ({ ...prev, image: "Please select a valid image file" }))
      return
    }

    // Validate file size (5MB limit)
    if (file.size > 5 * 1024 * 1024) {
      setErrors((prev) => ({ ...prev, image: "Image size must be less than 5MB" }))
      return
    }

    // Clear any previous image errors
    setErrors((prev) => ({ ...prev, image: "" }))

    // Clean up previous preview URL if it's a blob URL
    if (previewUrl && previewUrl.startsWith("blob:")) {
      URL.revokeObjectURL(previewUrl)
    }

    // Store file and create preview URL
    setFormData((prev) => ({ ...prev, file }))
    const newPreviewUrl = URL.createObjectURL(file)
    setPreviewUrl(newPreviewUrl)

    // Create object URL for preview
    const imageUrl = URL.createObjectURL(file)
    setFormData((prev) => ({ ...prev, imageUrl }))
  }

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      handleFileSelect(file)
    }
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(true)
  }

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)

    const file = e.dataTransfer.files[0]
    if (file) {
      handleFileSelect(file)
    }
  }

  const handleRemoveImage = () => {
    if (previewUrl && previewUrl.startsWith("blob:")) {
      URL.revokeObjectURL(previewUrl)
    }
    setFormData((prev) => ({ ...prev, file: null }))
    setPreviewUrl("")
    setFormData((prev) => ({ ...prev, imageUrl: "" }))
    if (fileInputRef.current) {
      fileInputRef.current.value = ""
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    setIsSubmitting(true)

    if (validateForm()) {
      try {
        await onUpdateProduct(formData)
      } catch (error) {
        toast.error("An error occurred while adding the product. Please try again.", {
          style: {
            background: "rgba(255, 0, 0, 0.1)",
            color: "#fff",
          },
        })
      } finally {
        setIsSubmitting(false)
      }
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

  const handleClose = () => {
    if (!isSubmitting) {
      // Clean up preview URL if it's a blob URL
      if (previewUrl && previewUrl.startsWith("blob:")) {
        URL.revokeObjectURL(previewUrl)
      }
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
      <div className="bg-gradient-to-b from-[#1a1a2e] to-[#16213e] rounded-xl border border-gray-700 w-full max-w-2xl shadow-2xl animate-in zoom-in-95 duration-300 max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-700 sticky top-0 bg-gradient-to-b from-[#1a1a2e] to-[#16213e] z-10">
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
        <form onSubmit={handleSubmit} className="p-6 space-y-6">
          {/* Image Upload Section */}
          <div className="space-y-4">
            <Label className="text-violet-400 font-medium">Product Image</Label>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleFileInputChange}
                  disabled={isSubmitting}
                  className="hidden"
                />
            {previewUrl ? (
              // Image Preview
              <div className="relative">
                <div className="w-full h-48 rounded-lg overflow-hidden bg-gray-800 border-2 border-gray-600">
                  <img
                    src={previewUrl || "/placeholder.svg"}
                    alt="Product preview"
                    className="w-full h-full object-cover"
                    crossOrigin="anonymous"
                  />
                </div>
                <Button
                  type="button"
                  variant="destructive"
                  size="sm"
                  onClick={handleRemoveImage}
                  disabled={isSubmitting}
                  className="absolute top-2 right-2 bg-red-500/80 hover:bg-red-500 backdrop-blur-sm"
                >
                  <Trash2 className="w-4 h-4" />
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={() => !isSubmitting && fileInputRef.current?.click()}
                  disabled={isSubmitting}
                  className="absolute bottom-2 right-2 bg-gray-800/80 hover:bg-gray-700 backdrop-blur-sm border-gray-600"
                >
                  <Upload className="w-4 h-4 mr-1" />
                  Change
                </Button>
              </div>
            ) : (
              // Upload Area
              <div
                className={`border-2 border-dashed rounded-lg p-8 text-center transition-all duration-300 cursor-pointer ${
                  isDragOver
                    ? "border-violet-400 bg-violet-500/10"
                    : "border-gray-600 hover:border-gray-500 hover:bg-gray-800/30"
                } ${isSubmitting ? "opacity-50 cursor-not-allowed" : ""}`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => !isSubmitting && fileInputRef.current?.click()}
              >
                <div className="flex flex-col items-center space-y-4">
                  <div className="p-4 rounded-full bg-gray-800">
                    {isDragOver ? (
                      <Upload className="w-8 h-8 text-violet-400" />
                    ) : (
                      <ImageIcon className="w-8 h-8 text-gray-400" />
                    )}
                  </div>
                  <div>
                    <p className="text-white font-medium mb-1">
                      {isDragOver ? "Drop image here" : "Upload product image"}
                    </p>
                    <p className="text-gray-400 text-sm">Drag and drop or click to browse (Max 5MB, JPG, PNG, GIF)</p>
                  </div>
                </div>
              </div>
            )}
            {errors.imageUrl && <p className="text-red-400 text-sm">{errors.imageUrl}</p>}
          </div>

          {/* Form Fields Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Left Column */}
            <div className="space-y-4">
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
          </div>

          {/* Right Column */}
          <div className="space-y-4">
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
            </div>
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
              className={`bg-[#0f0f23] border-gray-600 text-white focus:border-violet-400 focus:ring-violet-400/20 transition-all duration-300 min-h-[120px] resize-none disabled:opacity-50 ${
                errors.description ? "border-red-500" : ""
              }`}
              placeholder="Enter product description"
            />
            {errors.description && <p className="text-red-400 text-sm">{errors.description}</p>}
          </div>

          {/* Buttons */}
          <div className="flex space-x-3 pt-4 border-t border-gray-700">
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
              disabled={isSubmitting}
              className="flex-1 bg-gradient-to-r from-violet-500 to-purple-600 hover:from-violet-600 hover:to-purple-700 text-white shadow-lg hover:shadow-violet-500/25 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isSubmitting ? (
                <>
                  <div className="w-4 h-4 mr-2 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                  Updating Product...
                </>
              ) : (
                <>
                  <Edit className="w-4 h-4 mr-2" />
                  Update Product
                </>
              )}
            </Button>
          </div>
        </form>
      </div>
    </div>
  )

  // Render modal using portal to document.body
  return createPortal(modalContent, document.body)
}
