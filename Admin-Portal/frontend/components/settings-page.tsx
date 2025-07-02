"use client"

import { useState } from "react"
import { Save, MessageSquare, Shield, Palette } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Switch } from "@/components/ui/switch"

export function SettingsPage() {
  const [settings, setSettings] = useState({
    businessName: "Demo Business",
    email: "demo@business.com",
    whatsappNumber: "+1 (555) 123-4567",
    autoReply: true,
    notifications: true,
    darkMode: true,
    language: "English",
  })

  const handleSave = () => {
    // Save settings logic here
    console.log("Settings saved:", settings)
  }

  const updateSetting = (key: string, value: any) => {
    setSettings((prev) => ({ ...prev, [key]: value }))
  }

  return (
    <div className="space-y-6 animate-in fade-in-50 duration-500">
      <div>
        <h2 className="text-3xl font-bold text-white mb-2">Settings</h2>
        <p className="text-gray-400">Manage your account and application preferences</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Business Information */}
        <Card className="bg-gradient-to-br from-[#1a1a2e] to-[#16213e] border-gray-700 animate-in slide-in-from-left duration-700">
          <CardHeader>
            <CardTitle className="text-white flex items-center">
              <div className="p-2 rounded-lg bg-violet-500/10 mr-3">
                <Shield className="w-5 h-5 text-violet-400" />
              </div>
              Business Information
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="businessName" className="text-violet-400 font-medium">
                Business Name
              </Label>
              <Input
                id="businessName"
                value={settings.businessName}
                onChange={(e) => updateSetting("businessName", e.target.value)}
                className="bg-[#0f0f23] border-gray-600 text-white focus:border-violet-400 focus:ring-violet-400/20 transition-all duration-300"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="email" className="text-violet-400 font-medium">
                Email Address
              </Label>
              <Input
                id="email"
                type="email"
                value={settings.email}
                onChange={(e) => updateSetting("email", e.target.value)}
                className="bg-[#0f0f23] border-gray-600 text-white focus:border-violet-400 focus:ring-violet-400/20 transition-all duration-300"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="whatsappNumber" className="text-violet-400 font-medium">
                WhatsApp Number
              </Label>
              <Input
                id="whatsappNumber"
                value={settings.whatsappNumber}
                onChange={(e) => updateSetting("whatsappNumber", e.target.value)}
                className="bg-[#0f0f23] border-gray-600 text-white focus:border-violet-400 focus:ring-violet-400/20 transition-all duration-300"
              />
            </div>
          </CardContent>
        </Card>

        {/* Preferences */}
        <Card className="bg-gradient-to-br from-[#1a1a2e] to-[#16213e] border-gray-700 animate-in slide-in-from-right duration-700">
          <CardHeader>
            <CardTitle className="text-white flex items-center">
              <div className="p-2 rounded-lg bg-emerald-500/10 mr-3">
                <Palette className="w-5 h-5 text-emerald-400" />
              </div>
              Preferences
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
      </div>

      {/* Chat Settings */}
      <Card className="bg-gradient-to-br from-[#1a1a2e] to-[#16213e] border-gray-700 animate-in slide-in-from-bottom duration-700 delay-200">
        <CardHeader>
          <CardTitle className="text-white flex items-center">
            <div className="p-2 rounded-lg bg-violet-500/10 mr-3">
              <MessageSquare className="w-5 h-5 text-violet-400" />
            </div>
            Chat Settings
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <h4 className="text-emerald-400 font-semibold">Auto-Reply Messages</h4>
              <div className="space-y-2">
                <Label className="text-gray-300 text-sm">Welcome Message</Label>
                <Input
                  placeholder="Hi! Welcome to our store. How can I help you today?"
                  className="bg-[#0f0f23] border-gray-600 text-white focus:border-violet-400 focus:ring-violet-400/20 transition-all duration-300"
                />
              </div>
              <div className="space-y-2">
                <Label className="text-gray-300 text-sm">Away Message</Label>
                <Input
                  placeholder="We're currently away. We'll get back to you soon!"
                  className="bg-[#0f0f23] border-gray-600 text-white focus:border-violet-400 focus:ring-violet-400/20 transition-all duration-300"
                />
              </div>
            </div>
            <div className="space-y-4">
              <h4 className="text-emerald-400 font-semibold">Response Settings</h4>
              <div className="space-y-2">
                <Label className="text-gray-300 text-sm">Response Delay (seconds)</Label>
                <Input
                  type="number"
                  placeholder="2"
                  className="bg-[#0f0f23] border-gray-600 text-white focus:border-violet-400 focus:ring-violet-400/20 transition-all duration-300"
                />
              </div>
              <div className="space-y-2">
                <Label className="text-gray-300 text-sm">Max Messages per Hour</Label>
                <Input
                  type="number"
                  placeholder="100"
                  className="bg-[#0f0f23] border-gray-600 text-white focus:border-violet-400 focus:ring-violet-400/20 transition-all duration-300"
                />
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Save Button */}
      <div className="flex justify-end animate-in slide-in-from-bottom duration-700 delay-300">
        <Button
          onClick={handleSave}
          className="bg-gradient-to-r from-violet-500 to-purple-600 hover:from-violet-600 hover:to-purple-700 text-white px-8 py-3 shadow-lg hover:shadow-violet-500/25 transition-all duration-300"
        >
          <Save className="w-4 h-4 mr-2" />
          Save Settings
        </Button>
      </div>
    </div>
  )
}
