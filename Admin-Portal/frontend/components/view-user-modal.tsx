"use client"

import type React from "react"
import { useState, useEffect } from "react"
import { createPortal } from "react-dom"
import { X, Eye, Mail, Phone, MapPin, Calendar, ShoppingBag, DollarSign, Activity } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { useToast } from "@/hooks/use-toast"
import type { User } from "@/components/user-management-page"

interface ViewUserModalProps {
  isOpen: boolean
  onClose: () => void
  user: User | null
  onStatusChange: (userId: string, newStatus: "active" | "inactive" | "suspended") => void
}

export function ViewUserModal({ isOpen, onClose, user, onStatusChange }: ViewUserModalProps) {
  const [mounted, setMounted] = useState(false)
  const { toast } = useToast()

  useEffect(() => {
    setMounted(true)
  }, [])

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

  const getStatusColor = (status: string) => {
    switch (status) {
      case "active":
        return "bg-emerald-500/20 text-emerald-400 border-emerald-500/30"
      case "inactive":
        return "bg-yellow-500/20 text-yellow-400 border-yellow-500/30"
      case "suspended":
        return "bg-red-500/20 text-red-400 border-red-500/30"
      default:
        return "bg-gray-500/20 text-gray-400 border-gray-500/30"
    }
  }

  const getRoleColor = (role: string) => {
    switch (role) {
      case "admin":
        return "bg-purple-500/20 text-purple-400 border-purple-500/30"
      case "user":
        return "bg-blue-500/20 text-blue-400 border-blue-500/30"
      case "seller":
        return "bg-orange-500/20 text-orange-400 border-orange-500/30"
      default:
        return "bg-gray-500/20 text-gray-400 border-gray-500/30"
    }
  }

  const handleStatusChange = (newStatus: "active" | "inactive" | "suspended") => {
    if (user) {
      onStatusChange(user.id, newStatus)
    }
  }

  const handleBackdropClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) {
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
            <Eye className="w-5 h-5 mr-2 text-violet-400" />
            User Details
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

        {/* Content */}
        <div className="p-6 space-y-6">
          {/* User Header */}
          <div className="flex items-start justify-between">
            <div className="flex items-center space-x-4">
              <div className="w-16 h-16 bg-gradient-to-r from-violet-500 to-purple-600 rounded-full flex items-center justify-center text-white text-2xl font-bold">
                {user.businessName.charAt(0).toUpperCase()}
              </div>
              <div>
                <h3 className="text-2xl font-bold text-white">{user.businessName}</h3>
                <p className="text-gray-400">{user.email}</p>
                <div className="flex items-center space-x-2 mt-2">
                  <Badge className={getRoleColor(user.role)}>{user.role.toUpperCase()}</Badge>
                  <Badge className={getStatusColor(user.status)}>{user.status.toUpperCase()}</Badge>
                </div>
              </div>
            </div>
          </div>

          {/* Contact Information */}
          <div className="bg-[#0f0f23] rounded-lg p-4">
            <h4 className="text-lg font-semibold text-white mb-4 flex items-center">
              <Mail className="w-5 h-5 mr-2 text-violet-400" />
              Contact Information
            </h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="flex items-center space-x-3">
                <Mail className="w-4 h-4 text-gray-400" />
                <div>
                  <p className="text-gray-400 text-sm">Email</p>
                  <p className="text-white">{user.email}</p>
                </div>
              </div>
              <div className="flex items-center space-x-3">
                <Phone className="w-4 h-4 text-gray-400" />
                <div>
                  <p className="text-gray-400 text-sm">WhatsApp</p>
                  <p className="text-white">{user.whatsappNumber}</p>
                </div>
              </div>
              <div className="flex items-start space-x-3 md:col-span-2">
                <MapPin className="w-4 h-4 text-gray-400 mt-1" />
                <div>
                  <p className="text-gray-400 text-sm">Address</p>
                  <p className="text-white">{user.address}</p>
                </div>
              </div>
            </div>
          </div>

          {/* Account Information */}
          <div className="bg-[#0f0f23] rounded-lg p-4">
            <h4 className="text-lg font-semibold text-white mb-4 flex items-center">
              <Activity className="w-5 h-5 mr-2 text-emerald-400" />
              Account Information
            </h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="flex items-center space-x-3">
                <Calendar className="w-4 h-4 text-gray-400" />
                <div>
                  <p className="text-gray-400 text-sm">Member Since</p>
                  <p className="text-white">{user.createdAt}</p>
                </div>
              </div>
              <div className="flex items-center space-x-3">
                <Activity className="w-4 h-4 text-gray-400" />
                <div>
                  <p className="text-gray-400 text-sm">Last Login</p>
                  <p className="text-white">{user.lastLogin || "Never"}</p>
                </div>
              </div>
            </div>
          </div>

          {/* Business Metrics */}
          <div className="bg-[#0f0f23] rounded-lg p-4">
            <h4 className="text-lg font-semibold text-white mb-4 flex items-center">
              <ShoppingBag className="w-5 h-5 mr-2 text-violet-400" />
              Business Metrics
            </h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="text-center p-4 bg-[#1a1a2e] rounded-lg">
                <ShoppingBag className="w-8 h-8 text-emerald-400 mx-auto mb-2" />
                <p className="text-2xl font-bold text-emerald-400">{user.totalOrders}</p>
                <p className="text-gray-400 text-sm">Total Orders</p>
              </div>
              <div className="text-center p-4 bg-[#1a1a2e] rounded-lg">
                <DollarSign className="w-8 h-8 text-violet-400 mx-auto mb-2" />
                <p className="text-2xl font-bold text-violet-400">{user.totalEarned}</p>
                <p className="text-gray-400 text-sm">Total Spent</p>
              </div>
            </div>
          </div>

          {/* Status Management */}
          <div className="bg-[#0f0f23] rounded-lg p-4">
            <h4 className="text-lg font-semibold text-white mb-4">Status Management</h4>
            <div className="flex items-center space-x-4">
              <div className="flex-1">
                <p className="text-gray-400 text-sm mb-2">Change User Status</p>
                <Select
                  value={user.status}
                  onValueChange={handleStatusChange}
                >
                  <SelectTrigger className="bg-[#1a1a2e] border-gray-600 text-white focus:border-violet-400">
                    <SelectValue placeholder="Select status" />
                  </SelectTrigger>

                  <SelectContent
                    className="bg-[#1a1a2e] border-gray-600 z-[10000]"
                    side="bottom"
                    position="popper"
                  >
                    <SelectItem value="active" className="text-white hover:bg-gray-800">
                      <span className="flex items-center space-x-2">
                        <span className="w-2 h-2 bg-emerald-400 rounded-full" />
                        <span>Active</span>
                      </span>
                    </SelectItem>

                    <SelectItem value="inactive" className="text-white hover:bg-gray-800">
                      <span className="flex items-center space-x-2">
                        <span className="w-2 h-2 bg-yellow-400 rounded-full" />
                        <span>Inactive</span>
                      </span>
                    </SelectItem>

                    <SelectItem value="suspended" className="text-white hover:bg-gray-800">
                      <span className="flex items-center space-x-2">
                        <span className="w-2 h-2 bg-red-400 rounded-full" />
                        <span>Suspended</span>
                      </span>
                    </SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          </div>

          {/* Close Button */}
          <div className="flex justify-end pt-4 border-t border-gray-700">
            <Button
              onClick={onClose}
              className="bg-gradient-to-r from-violet-500 to-purple-600 hover:from-violet-600 hover:to-purple-700 text-white shadow-lg hover:shadow-violet-500/25 transition-all duration-300"
            >
              Close
            </Button>
          </div>
        </div>
      </div>
    </div>
  )

  return createPortal(modalContent, document.body)
}
