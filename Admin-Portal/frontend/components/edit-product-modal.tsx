"use client"

import type React from "react"

import { useState, useEffect, useRef } from "react"
import { createPortal } from "react-dom"
import { X, Edit, Upload, ImageIcon, Trash2, Star } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import type { Product } from "@/components/product-management"
import { toast } from "sonner"

interface EditableImage {
  item_image_id?: number
  file: File | null
  url: string
  isMain: boolean
  action: "keep" | "update" | "delete" | "add"
}

interface EditProductModalProps {
  isOpen: boolean
  onClose: () => void
  onUpdateProduct: (productData: {
    name: string
    price: number
    description: string
    stock: number
    imageUpdates: Array<{
      item_image_id?: number
      file?: File
      isMain: boolean
      action: "keep" | "update" | "delete" | "add"
      url?: string
    }>
    mainImageId?: number
  }) => Promise<void>
  product: Product | null
}

export function EditProductModal({ isOpen, onClose, onUpdateProduct, product }: EditProductModalProps) {
  const [mounted, setMounted] = useState(false)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const fileInputRefs = useRef<(HTMLInputElement | null)[]>([])
  const [formData, setFormData] = useState({
    name: "",
    price: 0,
    description: "",
    stock: 0,
  })

  const [images, setImages] = useState<EditableImage[]>([
    { file: null, url: "", isMain: true, action: "keep" },
    { file: null, url: "", isMain: false, action: "keep" },
    { file: null, url: "", isMain: false, action: "keep" },
    { file: null, url: "", isMain: false, action: "keep" },
    { file: null, url: "", isMain: false, action: "keep" },
  ])

  const [dragOverIndex, setDragOverIndex] = useState<number | null>(null)
  const [errors, setErrors] = useState({
    name: "",
    price: "",
    description: "",
    stock: "",
    images: "",
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

      // Initialize images from product data
      const productImages = product.images || []
      const initialImages: EditableImage[] = [
        { file: null, url: "", isMain: true, action: "keep" },
        { file: null, url: "", isMain: false, action: "keep" },
        { file: null, url: "", isMain: false, action: "keep" },
        { file: null, url: "", isMain: false, action: "keep" },
        { file: null, url: "", isMain: false, action: "keep" },
      ]

      // Fill with existing product images
      productImages.forEach((img, index) => {
        if (index < 5) {
          initialImages[index] = {
            item_image_id: img.id,
            file: null,
            url: img.url,
            isMain: img.isMain,
            action: "keep",
          }
        }
      })

      // Ensure we have a main image
      if (productImages.length > 0 && !productImages.some((img) => img.isMain)) {
        initialImages[0].isMain = true
      }

      setImages(initialImages)
      setErrors({
        name: "",
        price: "",
        description: "",
        stock: "",
        images: "",
      })
      setIsSubmitting(false)
      setDragOverIndex(null)
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
      images: "",
    }

    if (!formData.name.trim()) {
      newErrors.name = "Product name is required"
    }

    if (!formData.price) {
      newErrors.price = "Price is required"
    }

    if (formData.price < 0) {
      newErrors.price = "Price cannot be negative"
    }

    if (!formData.description.trim()) {
      newErrors.description = "Description is required"
    }

    if (formData.stock < 0) {
      newErrors.stock = "Stock cannot be negative"
    }

    // Check if at least one image exists (not deleted)
    const hasImages = images.some((img) => img.action !== "delete" && (img.file !== null || img.url !== ""))
    if (!hasImages) {
      newErrors.images = "At least one product image is required"
    }

    setErrors(newErrors)
    return !Object.values(newErrors).some((error) => error !== "")
  }

  const handleFileSelect = (file: File, index: number) => {
    // Validate file type
    if (!file.type.startsWith("image/")) {
      setErrors((prev) => ({ ...prev, images: "Please select valid image files only" }))
      return
    }

    // Validate file size (5MB limit)
    if (file.size > 5 * 1024 * 1024) {
      setErrors((prev) => ({ ...prev, images: "Each image must be less than 5MB" }))
      return
    }

    // Clear any previous image errors
    setErrors((prev) => ({ ...prev, images: "" }))

    // Clean up previous blob URL if it exists
    if (images[index].url && images[index].url.startsWith("blob:")) {
      URL.revokeObjectURL(images[index].url)
    }

    // Update the specific image slot
    const newPreviewUrl = URL.createObjectURL(file)
    setImages((prev) =>
      prev.map((img, i) => {
        if (i === index) {
          return {
            ...img,
            file,
            url: newPreviewUrl,
            action: img.item_image_id ? "update" : "add",
          }
        }
        return img
      }),
    )

    // Create object URL for preview
    const imageUrl = URL.createObjectURL(file)
    setFormData((prev) => ({ ...prev, imageUrl }))
  }

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>, index: number) => {
    const file = e.target.files?.[0]
    if (file) {
      handleFileSelect(file, index)
    }
  }

  const handleDragOver = (e: React.DragEvent, index: number) => {
    e.preventDefault()
    setDragOverIndex(index)
  }

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault()
    setDragOverIndex(null)
  }

  const handleDrop = (e: React.DragEvent, index: number) => {
    e.preventDefault()
    setDragOverIndex(null)

    const file = e.dataTransfer.files[0]
    if (file) {
      handleFileSelect(file, index)
    }
  }

  const handleRemoveImage = (index: number) => {
    // Clean up blob URL if it exists
    if (images[index].url && images[index].url.startsWith("blob:")) {
      URL.revokeObjectURL(images[index].url)
    }

    
    setImages((prev) =>
      prev.map((img, i) => {
        if (i === index) {
          if (img.item_image_id) {
            // Mark existing image for deletion
            return {
              ...img,
              file: null,
              url: "",
              action: "delete",
            }
          } else {
            // Remove new image that hasn't been saved yet
            return {
              ...img,
              file: null,
              url: "",
              action: "keep",
            }
          }
        }
        return img
      }),
    )

    if (fileInputRefs.current[index]) {
      fileInputRefs.current[index]!.value = ""
    }
  }

  const handleSetMainImage = (index: number) => {
    // Only allow setting main if this slot has an image and isn't marked for deletion
    if (images[index].action === "delete" || (!images[index].file && !images[index].url)) return

    setImages((prev) =>
      prev.map((img, i) => ({
        ...img,
        isMain: i === index,
      })),
    )
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!validateForm()) {
      toast.error("Validation Error !!", {
        style: {
          background: "rgba(255, 0, 0, 0.1)",
          color: "#fff",
        },
      })
      return
    }

    setIsSubmitting(true)

    try {
      // Find the main image ID
      const mainImage = images.find((img) => img.isMain && img.action !== "delete")
      const mainImageId = mainImage?.item_image_id

      // Prepare image updates for API
      const imageUpdates = images
        .filter(
          (img) =>
            img.action !== "keep" ||
            img.isMain !== (product?.images?.find((pImg) => pImg.id === img.item_image_id)?.isMain || false),
        )
        .map((img) => ({
          item_image_id: img.item_image_id,
          file: img.file || undefined,
          isMain: img.isMain,
          action: img.action,
          url: img.action === "keep" ? img.url : undefined,
        }))

      await onUpdateProduct({
        name: formData.name,
        price: formData.price,
        description: formData.description,
        stock: formData.stock,
        imageUpdates,
        mainImageId,
      })
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
      // Clean up blob URLs
      images.forEach((img) => {
        if (img.url && img.url.startsWith("blob:")) {
          URL.revokeObjectURL(img.url)
        }
      })
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
    <div className="bg-gradient-to-b from-[#1a1a2e] to-[#16213e] rounded-xl border border-gray-700 w-full max-w-4xl shadow-2xl animate-in zoom-in-95 duration-300 max-h-[90vh] overflow-y-auto">
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
          {/* Product Images Section */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <Label className="text-violet-400 font-medium">Product Images (Up to 5)</Label>
              <div className="text-sm text-gray-400">
                <Star className="w-4 h-4 inline mr-1 text-yellow-400" />
                Click the star to set main image
              </div>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
              {images.map((image, index) => (
                <div key={index} className="relative">
                  {/* Hidden file input */}
                  <input
                    ref={(el) => {fileInputRefs.current[index] = el}}
                    type="file"
                    accept="image/*"
                    onChange={(e) => handleFileInputChange(e, index)}
                    disabled={isSubmitting}
                    className="hidden"
                  />

              {image.url && image.action !== "delete" ? (
                // Image Preview
                    <div className="relative group">
                    <div
                      className={`w-full h-32 rounded-lg overflow-hidden bg-gray-800 border-2 ${
                        image.isMain ? "border-yellow-400" : "border-gray-600"
                      } ${image.action === ("delete" as typeof image.action) ? "opacity-50" : ""}`}
                    >
                      <img
                        src={image.url || "/placeholder.svg"}
                        alt={`Product image ${index + 1}`}
                        className="w-full h-full object-cover"
                        crossOrigin="anonymous"
                      />
                    </div>

                    {/* Main image indicator */}
                    {image.isMain && (
                      <div className="absolute top-1 left-1 bg-yellow-400 text-black px-1 py-0.5 rounded text-xs font-semibold">
                        MAIN
                      </div>
                    )}

                    {/* Update indicator */}
                    {image.action === "update" && (
                      <div className="absolute top-1 left-1 bg-blue-400 text-white px-1 py-0.5 rounded text-xs font-semibold">
                        UPDATED
                      </div>
                    )}

                    {/* Action buttons */}
                    <div className="absolute top-1 right-1 flex space-x-1">
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        onClick={() => handleSetMainImage(index)}
                        disabled={isSubmitting || image.isMain || image.action === ("delete" as typeof image.action)}
                        className={`w-6 h-6 p-0 ${
                          image.isMain
                            ? "bg-yellow-400/80 text-black"
                            : "bg-gray-800/80 hover:bg-yellow-400/80 hover:text-black text-yellow-400"
                        } backdrop-blur-sm`}
                      >
                        <Star className="w-3 h-3" />
                      </Button>
                <Button
                  type="button"
                  variant="destructive"
                  onClick={() => handleRemoveImage(index)}
                  disabled={isSubmitting}
                  className="w-6 h-6 p-0 bg-red-500/80 hover:bg-red-500 backdrop-blur-sm"
                >
                  <Trash2 className="w-3 h-3" />
                </Button>
              </div>

              {/* Change button */}
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={() => fileInputRefs.current[index]?.click()}
                disabled={isSubmitting}
                className="absolute bottom-1 left-1 right-1 bg-gray-800/80 hover:bg-gray-700 backdrop-blur-sm border-gray-600 text-xs h-6 disabled:opacity-50"
              >
                <Upload className="w-3 h-3 mr-1" />
                  Change
                </Button>
              </div>
            ) : (
              // Upload Area
              <div
              className={`w-full h-32 border-2 border-dashed rounded-lg flex flex-col items-center justify-center cursor-pointer transition-all duration-300 ${
                dragOverIndex === index
                  ? "border-violet-400 bg-violet-500/10"
                  : "border-gray-600 hover:border-gray-500 hover:bg-gray-800/30"
              } ${isSubmitting ? "opacity-50 cursor-not-allowed" : ""}`}
              onDragOver={(e) => handleDragOver(e, index)}
              onDragLeave={handleDragLeave}
              onDrop={(e) => handleDrop(e, index)}
              onClick={() => !isSubmitting && fileInputRefs.current[index]?.click()}
            >
              <div className="text-center">
                {dragOverIndex === index ? (
                  <Upload className="w-6 h-6 text-violet-400 mx-auto mb-1" />
                ) : (
                  <ImageIcon className="w-6 h-6 text-gray-400 mx-auto mb-1" />
                )}
                <p className="text-xs text-gray-400">{index === 0 ? "Main Image" : `Image ${index + 1}`}</p>
              </div>
            </div>
          )}
        </div>
      ))}
    </div>
    {errors.images && <p className="text-red-400 text-sm">{errors.images}</p>}
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
              type="number"
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
