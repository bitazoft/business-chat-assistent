"use client"

import type React from "react"
import { useState } from "react"
import { ArrowLeft, MessageCircle } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"

interface AuthPageProps {
  mode: "login" | "register"
  onLogin: (userData: { businessName: string; email: string }) => void
  onBack: () => void
  onSwitchMode: (mode: "login" | "register") => void
}

export function AuthPage({ mode, onLogin, onBack, onSwitchMode }: AuthPageProps) {
  const [formData, setFormData] = useState({
    businessName: "",
    email: "",
    whatsappNumber: "",
    password: "",
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    onLogin({
      businessName: formData.businessName || "Demo Business",
      email: formData.email,
    })
  }

  const handleInputChange = (field: string, value: string) => {
    setFormData((prev) => ({ ...prev, [field]: value }))
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#0f0f23] to-[#1a1a2e] flex items-center justify-center px-6">
      <div className="w-full max-w-md">
        {/* Back button */}
        <Button
          variant="ghost"
          onClick={onBack}
          className="mb-8 text-violet-400 hover:text-emerald-400 hover:bg-transparent transition-colors duration-300"
        >
          <ArrowLeft className="w-4 h-4 mr-2" />
          Back to Home
        </Button>

        {/* Form container */}
        <div className="bg-gradient-to-b from-[#1a1a2e] to-[#16213e] p-8 rounded-xl border border-gray-700 shadow-2xl">
          <div className="text-center mb-8">
            <MessageCircle className="w-12 h-12 text-violet-400 mx-auto mb-4" />
            <h2 className="text-3xl font-bold mb-2">{mode === "login" ? "Welcome Back" : "Create Account"}</h2>
            <p className="text-gray-400">
              {mode === "login"
                ? "Sign in to your WhatsApp Business dashboard"
                : "Start automating your WhatsApp Business today"}
            </p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-6">
            {mode === "register" && (
              <div className="space-y-2">
                <Label htmlFor="businessName" className="text-violet-400 font-medium">
                  Business Name
                </Label>
                <Input
                  id="businessName"
                  type="text"
                  value={formData.businessName}
                  onChange={(e) => handleInputChange("businessName", e.target.value)}
                  className="bg-[#0f0f23] border-gray-600 text-white focus:border-violet-400 focus:ring-violet-400/20 transition-all duration-300"
                  placeholder="Enter your business name"
                  required
                />
              </div>
            )}

            <div className="space-y-2">
              <Label htmlFor="email" className="text-violet-400 font-medium">
                Email Address
              </Label>
              <Input
                id="email"
                type="email"
                value={formData.email}
                onChange={(e) => handleInputChange("email", e.target.value)}
                className="bg-[#0f0f23] border-gray-600 text-white focus:border-violet-400 focus:ring-violet-400/20 transition-all duration-300"
                placeholder="Enter your email"
                required
              />
            </div>

            {mode === "register" && (
              <div className="space-y-2">
                <Label htmlFor="whatsappNumber" className="text-violet-400 font-medium">
                  WhatsApp Number
                </Label>
                <Input
                  id="whatsappNumber"
                  type="tel"
                  value={formData.whatsappNumber}
                  onChange={(e) => handleInputChange("whatsappNumber", e.target.value)}
                  className="bg-[#0f0f23] border-gray-600 text-white focus:border-violet-400 focus:ring-violet-400/20 transition-all duration-300"
                  placeholder="+1 (555) 123-4567"
                  required
                />
              </div>
            )}

            <div className="space-y-2">
              <Label htmlFor="password" className="text-violet-400 font-medium">
                Password
              </Label>
              <Input
                id="password"
                type="password"
                value={formData.password}
                onChange={(e) => handleInputChange("password", e.target.value)}
                className="bg-[#0f0f23] border-gray-600 text-white focus:border-violet-400 focus:ring-violet-400/20 transition-all duration-300"
                placeholder="Enter your password"
                required
              />
            </div>

            <Button
              type="submit"
              className="w-full bg-gradient-to-r from-violet-500 to-purple-600 hover:from-violet-600 hover:to-purple-700 text-white py-3 text-lg font-semibold shadow-lg hover:shadow-violet-500/25 transition-all duration-300"
            >
              {mode === "login" ? "Sign In" : "Create Account"}
            </Button>
          </form>

          <div className="text-center mt-6">
            <p className="text-gray-400">
              {mode === "login" ? "Don't have an account? " : "Already have an account? "}
              <button
                onClick={() => onSwitchMode(mode === "login" ? "register" : "login")}
                className="text-violet-400 hover:text-emerald-400 transition-colors duration-300 font-semibold"
              >
                {mode === "login" ? "Sign up" : "Sign in"}
              </button>
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
