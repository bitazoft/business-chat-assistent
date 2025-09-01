"use client"

import { useState, useEffect } from "react"
import { Save, Shield, Palette, Eye, EyeOff, Building2, MapPin, Phone, Mail, User, Lock } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Switch } from "@/components/ui/switch"
import { Textarea } from "@/components/ui/textarea"
import { getCurrentUser } from "@/lib/auth"
import businessService, { BusinessDetails, UpdateBusinessDetailsData, UpdatePasswordData } from "@/services/businessService"
import { toast } from "sonner"

export function SettingsPage() {
  const [businessDetails, setBusinessDetails] = useState<BusinessDetails | null>(null)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [showCurrentPassword, setShowCurrentPassword] = useState(false)
  const [showNewPassword, setShowNewPassword] = useState(false)
  const [showConfirmPassword, setShowConfirmPassword] = useState(false)
  
  // Form data state
  const [formData, setFormData] = useState({
    shopName: "",
    gstNumber: "",
    whatsappNumberId: "",
    ownerName: "",
    email: "",
    phone: "",
    address: ""
  })

  // Password change state
  const [passwordData, setPasswordData] = useState({
    currentPassword: "",
    newPassword: "",
    confirmPassword: ""
  })

  // Settings state
  const [settings, setSettings] = useState({
    autoReply: true,
    notifications: true,
    darkMode: true,
    language: "English",
  })

  const user = getCurrentUser()

  useEffect(() => {
    loadBusinessDetails()
  }, [])

  const loadBusinessDetails = async () => {
    if (!user?.sellerId) {
      toast.error("User not authenticated")
      setLoading(false)
      return
    }

    try {
      setLoading(true)
      const details = await businessService.getBusinessDetails(user.sellerId.toString())
      setBusinessDetails(details)
      setFormData({
        shopName: details.shopName || "",
        gstNumber: details.gstNumber || "",
        whatsappNumberId: details.whatsappNumberId || "",
        ownerName: details.ownerName || "",
        email: details.email || "",
        phone: details.phone || "",
        address: details.address || ""
      })
    } catch (error) {
      console.error("Error loading business details:", error)
    } finally {
      setLoading(false)
    }
  }

  const handleFormChange = (field: string, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }))
  }

  const handlePasswordChange = (field: string, value: string) => {
    setPasswordData(prev => ({ ...prev, [field]: value }))
  }

  const handleSaveBusinessDetails = async () => {
    if (!user?.sellerId) {
      toast.error("User not authenticated")
      return
    }

    // Validate required fields
    if (!formData.shopName || !formData.ownerName || !formData.email) {
      toast.error("Shop name, owner name, and email are required")
      return
    }

    try {
      setSaving(true)
      const updateData: UpdateBusinessDetailsData = {
        shopName: formData.shopName,
        gstNumber: formData.gstNumber,
        whatsappNumberId: formData.whatsappNumberId,
        ownerName: formData.ownerName,
        email: formData.email,
        phone: formData.phone,
        address: formData.address
      }

      const updatedDetails = await businessService.updateBusinessDetails(user.sellerId.toString(), updateData)
      setBusinessDetails(updatedDetails)
      
    } catch (error) {
      console.error("Error updating business details:", error)
    } finally {
      setSaving(false)
    }
  }

  const handleChangePassword = async () => {
    if (!user?.sellerId) {
      toast.error("User not authenticated")
      return
    }

    // Validate password fields
    if (!passwordData.currentPassword || !passwordData.newPassword || !passwordData.confirmPassword) {
      toast.error("All password fields are required")
      return
    }

    if (passwordData.newPassword !== passwordData.confirmPassword) {
      toast.error("New password and confirm password don't match")
      return
    }

    if (passwordData.newPassword.length < 6) {
      toast.error("New password must be at least 6 characters long")
      return
    }

    try {
      setSaving(true)
      const updateData: UpdatePasswordData = {
        currentPassword: passwordData.currentPassword,
        newPassword: passwordData.newPassword
      }

      await businessService.updateBusinessPassword(user.sellerId.toString(), updateData)
      
      // Clear password fields
      setPasswordData({
        currentPassword: "",
        newPassword: "",
        confirmPassword: ""
      })
      
    } catch (error) {
      console.error("Error updating password:", error)
    } finally {
      setSaving(false)
    }
  }

  const updateSetting = (key: string, value: any) => {
    setSettings((prev) => ({ ...prev, [key]: value }))
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-white">Loading business details...</div>
      </div>
    )
  }

  return (
    <div className="space-y-6 animate-in fade-in-50 duration-500">
      <div>
        <h2 className="text-3xl font-bold text-white mb-2">Business Settings</h2>
        <p className="text-gray-400">Manage your business information and preferences</p>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        {/* Business Information */}
        <Card className="bg-gradient-to-br from-[#1a1a2e] to-[#16213e] border-gray-700 animate-in slide-in-from-left duration-700">
          <CardHeader>
            <CardTitle className="text-white flex items-center">
              <div className="p-2 rounded-lg bg-violet-500/10 mr-3">
                <Building2 className="w-5 h-5 text-violet-400" />
              </div>
              Business Information
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="shopName" className="text-violet-400 font-medium flex items-center">
                <Building2 className="w-4 h-4 mr-2" />
                Business Name *
              </Label>
              <Input
                id="shopName"
                value={formData.shopName}
                onChange={(e) => handleFormChange("shopName", e.target.value)}
                className="bg-[#0f0f23] border-gray-600 text-white focus:border-violet-400 focus:ring-violet-400/20 transition-all duration-300"
                placeholder="Enter your business name"
              />
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="gstNumber" className="text-violet-400 font-medium">
                GST Number
              </Label>
              <Input
                id="gstNumber"
                value={formData.gstNumber}
                onChange={(e) => handleFormChange("gstNumber", e.target.value)}
                className="bg-[#0f0f23] border-gray-600 text-white focus:border-violet-400 focus:ring-violet-400/20 transition-all duration-300"
                placeholder="Enter GST number (optional)"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="whatsappNumberId" className="text-violet-400 font-medium flex items-center">
                <Phone className="w-4 h-4 mr-2" />
                WhatsApp Business Number ID
              </Label>
              <Input
                id="whatsappNumberId"
                value={formData.whatsappNumberId}
                onChange={(e) => handleFormChange("whatsappNumberId", e.target.value)}
                className="bg-[#0f0f23] border-gray-600 text-white focus:border-violet-400 focus:ring-violet-400/20 transition-all duration-300"
                placeholder="Enter WhatsApp number ID"
              />
            </div>
          </CardContent>
        </Card>

        {/* Owner Information */}
        <Card className="bg-gradient-to-br from-[#1a1a2e] to-[#16213e] border-gray-700 animate-in slide-in-from-right duration-700">
          <CardHeader>
            <CardTitle className="text-white flex items-center">
              <div className="p-2 rounded-lg bg-emerald-500/10 mr-3">
                <User className="w-5 h-5 text-emerald-400" />
              </div>
              Owner Information
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="ownerName" className="text-emerald-400 font-medium flex items-center">
                <User className="w-4 h-4 mr-2" />
                Owner Name *
              </Label>
              <Input
                id="ownerName"
                value={formData.ownerName}
                onChange={(e) => handleFormChange("ownerName", e.target.value)}
                className="bg-[#0f0f23] border-gray-600 text-white focus:border-emerald-400 focus:ring-emerald-400/20 transition-all duration-300"
                placeholder="Enter owner name"
              />
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="email" className="text-emerald-400 font-medium flex items-center">
                <Mail className="w-4 h-4 mr-2" />
                Email Address *
              </Label>
              <Input
                id="email"
                type="email"
                value={formData.email}
                onChange={(e) => handleFormChange("email", e.target.value)}
                className="bg-[#0f0f23] border-gray-600 text-white focus:border-emerald-400 focus:ring-emerald-400/20 transition-all duration-300"
                placeholder="Enter email address"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="phone" className="text-emerald-400 font-medium flex items-center">
                <Phone className="w-4 h-4 mr-2" />
                Phone Number
              </Label>
              <Input
                id="phone"
                value={formData.phone}
                onChange={(e) => handleFormChange("phone", e.target.value)}
                className="bg-[#0f0f23] border-gray-600 text-white focus:border-emerald-400 focus:ring-emerald-400/20 transition-all duration-300"
                placeholder="Enter phone number"
              />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Address */}
      <Card className="bg-gradient-to-br from-[#1a1a2e] to-[#16213e] border-gray-700 animate-in slide-in-from-bottom duration-700 delay-100">
        <CardHeader>
          <CardTitle className="text-white flex items-center">
            <div className="p-2 rounded-lg bg-blue-500/10 mr-3">
              <MapPin className="w-5 h-5 text-blue-400" />
            </div>
            Business Address
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            <Label htmlFor="address" className="text-blue-400 font-medium flex items-center">
              <MapPin className="w-4 h-4 mr-2" />
              Complete Address
            </Label>
            <Textarea
              id="address"
              value={formData.address}
              onChange={(e) => handleFormChange("address", e.target.value)}
              className="bg-[#0f0f23] border-gray-600 text-white focus:border-blue-400 focus:ring-blue-400/20 transition-all duration-300"
              placeholder="Enter complete business address"
              rows={3}
            />
          </div>
        </CardContent>
      </Card>

      {/* Password Change */}
      <Card className="bg-gradient-to-br from-[#1a1a2e] to-[#16213e] border-gray-700 animate-in slide-in-from-bottom duration-700 delay-200">
        <CardHeader>
          <CardTitle className="text-white flex items-center">
            <div className="p-2 rounded-lg bg-red-500/10 mr-3">
              <Lock className="w-5 h-5 text-red-400" />
            </div>
            Change Password
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="space-y-2">
              <Label htmlFor="currentPassword" className="text-red-400 font-medium">
                Current Password
              </Label>
              <div className="relative">
                <Input
                  id="currentPassword"
                  type={showCurrentPassword ? "text" : "password"}
                  value={passwordData.currentPassword}
                  onChange={(e) => handlePasswordChange("currentPassword", e.target.value)}
                  className="bg-[#0f0f23] border-gray-600 text-white focus:border-red-400 focus:ring-red-400/20 transition-all duration-300 pr-10"
                  placeholder="Enter current password"
                />
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  className="absolute right-1 top-1 h-8 w-8 p-0"
                  onClick={() => setShowCurrentPassword(!showCurrentPassword)}
                >
                  {showCurrentPassword ? (
                    <EyeOff className="h-4 w-4 text-gray-400" />
                  ) : (
                    <Eye className="h-4 w-4 text-gray-400" />
                  )}
                </Button>
              </div>
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="newPassword" className="text-red-400 font-medium">
                New Password
              </Label>
              <div className="relative">
                <Input
                  id="newPassword"
                  type={showNewPassword ? "text" : "password"}
                  value={passwordData.newPassword}
                  onChange={(e) => handlePasswordChange("newPassword", e.target.value)}
                  className="bg-[#0f0f23] border-gray-600 text-white focus:border-red-400 focus:ring-red-400/20 transition-all duration-300 pr-10"
                  placeholder="Enter new password"
                />
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  className="absolute right-1 top-1 h-8 w-8 p-0"
                  onClick={() => setShowNewPassword(!showNewPassword)}
                >
                  {showNewPassword ? (
                    <EyeOff className="h-4 w-4 text-gray-400" />
                  ) : (
                    <Eye className="h-4 w-4 text-gray-400" />
                  )}
                </Button>
              </div>
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="confirmPassword" className="text-red-400 font-medium">
                Confirm Password
              </Label>
              <div className="relative">
                <Input
                  id="confirmPassword"
                  type={showConfirmPassword ? "text" : "password"}
                  value={passwordData.confirmPassword}
                  onChange={(e) => handlePasswordChange("confirmPassword", e.target.value)}
                  className="bg-[#0f0f23] border-gray-600 text-white focus:border-red-400 focus:ring-red-400/20 transition-all duration-300 pr-10"
                  placeholder="Confirm new password"
                />
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  className="absolute right-1 top-1 h-8 w-8 p-0"
                  onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                >
                  {showConfirmPassword ? (
                    <EyeOff className="h-4 w-4 text-gray-400" />
                  ) : (
                    <Eye className="h-4 w-4 text-gray-400" />
                  )}
                </Button>
              </div>
            </div>
          </div>
          
          <div className="flex justify-end">
            <Button
              onClick={handleChangePassword}
              disabled={saving || !passwordData.currentPassword || !passwordData.newPassword || !passwordData.confirmPassword}
              className="bg-gradient-to-r from-red-500 to-red-600 hover:from-red-600 hover:to-red-700 text-white px-6 py-2"
            >
              {saving ? "Updating..." : "Change Password"}
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Preferences */}
      <Card className="bg-gradient-to-br from-[#1a1a2e] to-[#16213e] border-gray-700 animate-in slide-in-from-bottom duration-700 delay-300">
        <CardHeader>
          <CardTitle className="text-white flex items-center">
            <div className="p-2 rounded-lg bg-emerald-500/10 mr-3">
              <Palette className="w-5 h-5 text-emerald-400" />
            </div>
            Application Preferences
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="flex items-center justify-between p-3 bg-[#0f0f23] rounded-lg">
            <div>
              <Label className="text-white font-medium">Auto Reply</Label>
              <p className="text-gray-400 text-sm">Automatically respond to customer messages</p>
            </div>
            <Switch
              checked={settings.autoReply}
              onCheckedChange={(checked) => updateSetting("autoReply", checked)}
              className="data-[state=checked]:bg-violet-500"
            />
          </div>
          <div className="flex items-center justify-between p-3 bg-[#0f0f23] rounded-lg">
            <div>
              <Label className="text-white font-medium">Notifications</Label>
              <p className="text-gray-400 text-sm">Receive push notifications for new messages</p>
            </div>
            <Switch
              checked={settings.notifications}
              onCheckedChange={(checked) => updateSetting("notifications", checked)}
              className="data-[state=checked]:bg-violet-500"
            />
          </div>
          <div className="flex items-center justify-between p-3 bg-[#0f0f23] rounded-lg">
            <div>
              <Label className="text-white font-medium">Dark Mode</Label>
              <p className="text-gray-400 text-sm">Use dark theme for the interface</p>
            </div>
            <Switch
              checked={settings.darkMode}
              onCheckedChange={(checked) => updateSetting("darkMode", checked)}
              className="data-[state=checked]:bg-violet-500"
            />
          </div>
        </CardContent>
      </Card>

      {/* Save Button */}
      <div className="flex justify-end animate-in slide-in-from-bottom duration-700 delay-400">
        <Button
          onClick={handleSaveBusinessDetails}
          disabled={saving}
          className="bg-gradient-to-r from-violet-500 to-purple-600 hover:from-violet-600 hover:to-purple-700 text-white px-8 py-3 shadow-lg hover:shadow-violet-500/25 transition-all duration-300"
        >
          <Save className="w-4 h-4 mr-2" />
          {saving ? "Saving..." : "Save Business Details"}
        </Button>
      </div>
    </div>
  )
}
