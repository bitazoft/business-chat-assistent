"use client"

import type React from "react"
import { useState, useEffect } from "react"
import { createPortal } from "react-dom"
import { X, Edit } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Textarea } from "@/components/ui/textarea"
import { useToast } from "@/hooks/use-toast"
import type { User } from "@/components/user-management-page"

interface EditUserModalProps {
  isOpen: boolean
  onClose: () => void
  onUpdateUser: (userData: User) => Promise<void>
  user: User | null
}

export function EditUserModal({ isOpen, onClose, onUpdateUser, user }: EditUserModalProps) {
  const [mounted, setMounted] = useState(false)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [formData, setFormData] = useState({
    businessName: "",
    email: "",
    whatsappNumber: "",
    address: "",
    role: "user" as "admin" | "user",
    status: "active" as "active" | "inactive" | "suspended",
  })

  const [errors, setErrors] = useState({
    businessName: "",
    email: "",
    whatsappNumber: "",
    address: "",
  })

  const { toast } = useToast()

  useEffect(() => {
    setMounted(true)
  }, [])

  useEffect(() => {
    if (isOpen && user) {
      setFormData({
        businessName: user.businessName,
        email: user.email,
        whatsappNumber: user.whatsappNumber,
        address: user.address,
        role: user.role,
        status: user.status,
      })
      setErrors({
        businessName: "",
        email: "",
        whatsappNumber: "",
        address: "",
      })
      setIsSubmitting(false)
    }
  }, [isOpen, user])

  // Blur effect
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = "hidden"
      const mainContent = document.querySelector("main")
      const topbar = document.querySelector("header")
      if (mainContent) mainContent.classList.add("blur-sm")
      if (topbar) topbar.classList.add("blur-sm")
    } else {
      document.body.style.overflow = "unset"
      const mainContent = document.querySelector("main")
      const topbar = document.querySelector("header")
      if (mainContent) mainContent.classList.remove("blur-sm")
      if (topbar) topbar.classList.remove("blur-sm")
    }

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
      businessName: "",
      email: "",
      whatsappNumber: "",
      address: "",
    }

    if (!formData.businessName.trim()) {
      newErrors.businessName = "Business name is required"
    }

    if (!formData.email.trim()) {
      newErrors.email = "Email is required"
    } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(formData.email)) {
      newErrors.email = "Please enter a valid email address"
    }

    if (!formData.whatsappNumber.trim()) {
      newErrors.whatsappNumber = "WhatsApp number is required"
    }

    if (!formData.address.trim()) {
      newErrors.address = "Address is required"
    }

    setErrors(newErrors)
    return !Object.values(newErrors).some((error) => error !== "")
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!validateForm() || !user) {
      toast({
        title: "Validation Error",
        description: "Please fix the errors in the form before submitting",
        variant: "destructive",
      })
      return
    }

    setIsSubmitting(true)

    try {
      const updatedUser: User = {
        ...user,
        ...formData,
      }
      await onUpdateUser(updatedUser)
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to update user. Please try again.",
        variant: "destructive",
      })
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleInputChange = (field: string, value: string) => {
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
    if (e.target === e.currentTarget && !isSubmitting) {
      onClose()
    }
  }

  if (!mounted || !isOpen || !user) return null

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
            Edit User
          </h2>
          <Button
            variant="ghost"
            size="icon"
            onClick={onClose}
            disabled={isSubmitting}
            className="text-gray-400 hover:text-white hover:bg-gray-800/50 transition-colors duration-200"
          >
            <X className="w-5 h-5" />
          </Button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="p-6 space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Business Name */}
            <div className="space-y-2">
              <Label htmlFor="editBusinessName" className="text-violet-400 font-medium">
                Business Name *
              </Label>
              <Input
                id="editBusinessName"
                type="text"
                value={formData.businessName}
                onChange={(e) => handleInputChange("businessName", e.target.value)}
                disabled={isSubmitting}
                className={`bg-[#0f0f23] border-gray-600 text-white focus:border-violet-400 focus:ring-violet-400/20 transition-all duration-300 ${
                  errors.businessName ? "border-red-500" : ""
                }`}
                placeholder="Enter business name"
                autoFocus
              />
              {errors.businessName && <p className="text-red-400 text-sm">{errors.businessName}</p>}
            </div>

            {/* Email */}
            <div className="space-y-2">
              <Label htmlFor="editEmail" className="text-violet-400 font-medium">
                Email Address *
              </Label>
              <Input
                id="editEmail"
                type="email"
                value={formData.email}
                onChange={(e) => handleInputChange("email", e.target.value)}
                disabled={isSubmitting}
                className={`bg-[#0f0f23] border-gray-600 text-white focus:border-violet-400 focus:ring-violet-400/20 transition-all duration-300 ${
                  errors.email ? "border-red-500" : ""
                }`}
                placeholder="Enter email address"
              />
              {errors.email && <p className="text-red-400 text-sm">{errors.email}</p>}
            </div>

            {/* WhatsApp Number */}
            <div className="space-y-2">
              <Label htmlFor="editWhatsappNumber" className="text-violet-400 font-medium">
                WhatsApp Number *
              </Label>
              <Input
                id="editWhatsappNumber"
                type="tel"
                value={formData.whatsappNumber}
                onChange={(e) => handleInputChange("whatsappNumber", e.target.value)}
                disabled={isSubmitting}
                className={`bg-[#0f0f23] border-gray-600 text-white focus:border-violet-400 focus:ring-violet-400/20 transition-all duration-300 ${
                  errors.whatsappNumber ? "border-red-500" : ""
                }`}
                placeholder="+1 (555) 123-4567"
              />
              {errors.whatsappNumber && <p className="text-red-400 text-sm">{errors.whatsappNumber}</p>}
            </div>

            {/* Role */}
            <div className="space-y-2">
              <Label className="text-violet-400 font-medium">Role *</Label>
              <Select
                value={formData.role}
                onValueChange={(value: "admin" | "user") => handleInputChange("role", value)}
              >
                <SelectTrigger className="bg-[#0f0f23] border-gray-600 text-white focus:border-violet-400">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="bg-[#1a1a2e] border-gray-600">
                  <SelectItem value="user" className="text-white hover:bg-gray-800">
                    User
                  </SelectItem>
                  <SelectItem value="admin" className="text-white hover:bg-gray-800">
                    Admin
                  </SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          {/* Address */}
          <div className="space-y-2">
            <Label htmlFor="editAddress" className="text-violet-400 font-medium">
              Business Address *
            </Label>
            <Textarea
              id="editAddress"
              value={formData.address}
              onChange={(e) => handleInputChange("address", e.target.value)}
              disabled={isSubmitting}
              className={`bg-[#0f0f23] border-gray-600 text-white focus:border-violet-400 focus:ring-violet-400/20 transition-all duration-300 min-h-[80px] resize-none ${
                errors.address ? "border-red-500" : ""
              }`}
              placeholder="Enter business address"
            />
            {errors.address && <p className="text-red-400 text-sm">{errors.address}</p>}
          </div>

          {/* Status */}
          <div className="space-y-2">
            <Label className="text-violet-400 font-medium">Status</Label>
            <Select
              value={formData.status}
              onValueChange={(value: "active" | "inactive" | "suspended") => handleInputChange("status", value)}
            >
              <SelectTrigger className="bg-[#0f0f23] border-gray-600 text-white focus:border-violet-400">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="bg-[#1a1a2e] border-gray-600">
                <SelectItem value="active" className="text-white hover:bg-gray-800">
                  Active
                </SelectItem>
                <SelectItem value="inactive" className="text-white hover:bg-gray-800">
                  Inactive
                </SelectItem>
                <SelectItem value="suspended" className="text-white hover:bg-gray-800">
                  Suspended
                </SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Buttons */}
          <div className="flex space-x-3 pt-4 border-t border-gray-700">
            <Button
              type="button"
              variant="outline"
              onClick={onClose}
              disabled={isSubmitting}
              className="flex-1 border-gray-600 text-gray-300 hover:text-white hover:border-gray-500 bg-transparent"
            >
              Cancel
            </Button>
            <Button
              type="submit"
              disabled={isSubmitting}
              className="flex-1 bg-gradient-to-r from-violet-500 to-purple-600 hover:from-violet-600 hover:to-purple-700 text-white shadow-lg hover:shadow-violet-500/25 transition-all duration-300"
            >
              {isSubmitting ? (
                <>
                  <div className="w-4 h-4 mr-2 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                  Updating User...
                </>
              ) : (
                <>
                  <Edit className="w-4 h-4 mr-2" />
                  Update User
                </>
              )}
            </Button>
          </div>
        </form>
      </div>
    </div>
  )

  return createPortal(modalContent, document.body)
}
