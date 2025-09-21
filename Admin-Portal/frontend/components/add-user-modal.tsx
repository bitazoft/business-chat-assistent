"use client"

import type React from "react"
import { useState, useEffect } from "react"
import { createPortal } from "react-dom"
import { X, UserPlus } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { SignupFields } from "@/components/signup-fields"
import { toast } from "sonner"

interface AddUserModalProps {
  isOpen: boolean
  onClose: () => void
  onAddUser: (userData: {
    shop_name: string
    email: string
    password: string
    phone: string
    address: string
    name: string
    whatsapp_number_id: string
    role: string
  }) => Promise<void>
}

export function AddUserModal({ isOpen, onClose, onAddUser }: AddUserModalProps) {
  const [mounted, setMounted] = useState(false)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [formData, setFormData] = useState({
    shop_name: "",
    email: "",
    phone: "",
    address: "",
    name: "",
    whatsapp_number_id: "",
    password: "",
    confirmPassword: "",
    role: "seller" as "seller" | "admin",
  })

  const [errors, setErrors] = useState({
    shop_name: "",
    email: "",
    phone: "",
    address: "",
    name: "",
    whatsapp_number_id: "",
    password: "",
    confirmPassword: "",
  })
  const [passwordsMatch, setPasswordsMatch] = useState(true)
  const [showPasswordError, setShowPasswordError] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  useEffect(() => {
    if (isOpen) {
      setFormData({
        shop_name: "",
        email: "",
        phone: "",
        address: "",
        name: "",
        whatsapp_number_id: "",
        password: "",
        confirmPassword: "",
        role: "seller",
      })
      setErrors({
        shop_name: "",
        email: "",
        phone: "",
        address: "",
        name: "",
        whatsapp_number_id: "",
        password: "",
        confirmPassword: "",
      })
      setPasswordsMatch(true)
      setShowPasswordError(false)
      setIsSubmitting(false)
    }
  }, [isOpen])

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
      shop_name: "",
      email: "",
      phone: "",
      address: "",
      name: "",
      whatsapp_number_id: "",
      password: "",
      confirmPassword: "",
    }

    if (!formData.shop_name.trim()) newErrors.shop_name = "Business name is required"
    if (!formData.name.trim()) newErrors.name = "Owner name is required"
    if (!formData.email.trim()) newErrors.email = "Email is required"
    else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(formData.email)) newErrors.email = "Please enter a valid email address"
    if (!formData.phone.trim()) newErrors.phone = "WhatsApp number is required"
    if (!formData.address.trim()) newErrors.address = "Address is required"
    if (!formData.whatsapp_number_id.trim()) newErrors.whatsapp_number_id = "WhatsApp number ID is required"
    if (!formData.password.trim()) newErrors.password = "Password is required"
    if (!formData.confirmPassword.trim()) newErrors.confirmPassword = "Confirm password is required"
    if (formData.password && formData.confirmPassword && formData.password !== formData.confirmPassword) newErrors.confirmPassword = "Passwords do not match"

    setErrors(newErrors)
    setPasswordsMatch(formData.password === formData.confirmPassword)
    setShowPasswordError(!!newErrors.confirmPassword)
    return !Object.values(newErrors).some((error) => error !== "")
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!validateForm()) {
      toast.error( "Validation Error",{
        description: "Please fix the errors in the form before submitting",
      })
      return
    }

    setIsSubmitting(true)

    try {
      await onAddUser({
        shop_name: formData.shop_name,
        email: formData.email,
        password: formData.password,
        phone: formData.phone,
        address: formData.address,
        name: formData.name,
        whatsapp_number_id: formData.whatsapp_number_id,
        role: formData.role,
      })

      toast.success("User added successfully")
    } catch (error) {
      toast.error("Error",{
        description: "Failed to add user. Please try again.",
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

    if ((errors as any)[field]) {
      setErrors((prev) => ({
        ...prev,
        [field]: "",
      }))
    }

    if (field === "password" || field === "confirmPassword") {
      const password = field === "password" ? value : formData.password
      const confirmPassword = field === "confirmPassword" ? value : formData.confirmPassword
      setPasswordsMatch(password === confirmPassword)
      setShowPasswordError(confirmPassword.length > 0 && password !== confirmPassword)
    }
  }

  const handleBackdropClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget && !isSubmitting) {
      onClose()
    }
  }

  if (!mounted || !isOpen) return null

  const modalContent = (
    <div
      className="fixed inset-0 bg-black/60 flex items-center justify-center z-[9999] p-4 animate-in fade-in-0 duration-300"
      onClick={handleBackdropClick}
    >
      <div className="bg-gradient-to-b from-[#1a1a2e] to-[#16213e] rounded-xl border border-gray-700 w-full max-w-2xl shadow-2xl animate-in zoom-in-95 duration-300 max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-700 sticky top-0 bg-gradient-to-b from-[#1a1a2e] to-[#16213e] z-10">
          <h2 className="text-xl font-bold text-white flex items-center">
            <UserPlus className="w-5 h-5 mr-2 text-violet-400" />
            Add New User
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
          <SignupFields
            formData={formData}
            onChange={handleInputChange}
            passwordsMatch={passwordsMatch}
            showPasswordError={showPasswordError}
          />

          {/* Role */}
        <div className="space-y-2">
            <Label className="text-violet-400 font-medium">Role *</Label>
            <Select
                value={formData.role}
                onValueChange={(value: "seller" | "admin") => handleInputChange("role", value)}
                >
                <SelectTrigger className="bg-[#0f0f23] border-gray-600 text-white focus:border-violet-400">
                    <SelectValue placeholder="Select role" />
                </SelectTrigger>
                <SelectContent
                    className="bg-[#1a1a2e] border-gray-600 z-[10000]"
                    side="bottom"
                    position="popper"
                >
                    <SelectItem value="seller" className="text-white hover:bg-gray-800">
                    Seller
                    </SelectItem>
                    <SelectItem value="admin" className="text-white hover:bg-gray-800">
                    Admin
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
                  Creating User...
                </>
              ) : (
                <>
                  <UserPlus className="w-4 h-4 mr-2" />
                  Add User
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
