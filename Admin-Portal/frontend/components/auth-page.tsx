"use client"

import type React from "react"
import { useState } from "react"
import { ArrowLeft, MessageCircle, Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { toast } from "sonner"

interface AuthPageProps {
  mode: "login" | "register"
  onLogin: (userData: { id: number; name: string; email: string; role: string; sellerId: number | null}) => void
  onBack: () => void
  onSwitchMode: (mode: "login" | "register") => void
}

export function AuthPage({ mode, onLogin, onBack }: AuthPageProps) {
  const [formData, setFormData] = useState({
    email: "",
    password: "",
  })

  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")
  const [success, setSuccess] = useState("")

  // Login logic only
  const loginLogic = async (credentials: { email: string; password: string }) => {
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_BASE_URL}/auth/login`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(credentials),
      })

      const data = await response.json()

      if (response.ok) {
        return { success: true, user: data.user }
      } else {
        return { success: false, message: data.message || "Login failed" }
      }
    } catch (error) {
      return { success: false, message: "Network error occurred" }
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError("")
    setSuccess("")

    try {
      const loginResult = await loginLogic({
        email: formData.email,
        password: formData.password,
      })

      if (loginResult.success) {
        onLogin(loginResult.user)
      } else {
        throw new Error(loginResult.message || "Invalid credentials")
      }
    } catch (error: any) {
      toast.error(error.message || "Authentication failed")
    } finally {
      setLoading(false)
    }
  }

  const handleInputChange = (field: string, value: string) => {
    setFormData((prev) => ({ ...prev, [field]: value }))
  }

  return (
    <div className="h-screen bg-gradient-to-br from-[#0f0f23] to-[#1a1a2e] flex items-center justify-center px-6 overflow-hidden">
      <div className={`w-full max-w-md`}>
        {/* Back button */}
        <Button
          variant="ghost"
          onClick={onBack}
          className="mb-3 text-violet-400 hover:text-emerald-400 hover:bg-transparent transition-colors duration-300"
          disabled={loading}
        >
          <ArrowLeft className="w-4 h-4 mr-2" />
          Back to Home
        </Button>

        {/* Form container */}
        <div className="bg-gradient-to-b from-[#1a1a2e] to-[#16213e] p-6 rounded-xl border border-gray-700 shadow-2xl">
          {/* Header */}
          <div className="text-center mb-4">
            <MessageCircle className="w-8 h-8 text-violet-400 mx-auto mb-2" />
            <h2 className="text-xl font-bold mb-1">Welcome Back</h2>
            <p className="text-gray-400 text-sm">Sign in to your WhatsApp Business dashboard</p>
          </div>

          {/* Error/Success Messages */}
          {error && (
            <Alert className="mb-4 border-red-500/50 bg-red-500/10">
              <AlertDescription className="text-red-400">{error}</AlertDescription>
            </Alert>
          )}

          {success && (
            <Alert className="mb-4 border-emerald-500/50 bg-emerald-500/10">
              <AlertDescription className="text-emerald-400">{success}</AlertDescription>
            </Alert>
          )}

          <form onSubmit={handleSubmit}>
            <div className="space-y-3 mb-4">
              <div className="space-y-1">
                <Label htmlFor="email" className="text-violet-400 font-medium text-sm">
                  Email Address
                </Label>
                <Input
                  id="email"
                  type="email"
                  value={formData.email}
                  onChange={(e) => handleInputChange("email", e.target.value)}
                  className="bg-[#0f0f23] border-gray-600 text-white focus:border-violet-400 focus:ring-violet-400/20 transition-all duration-300 h-10"
                  placeholder="Enter your email"
                  required
                />
              </div>

              <div className="space-y-1">
                <Label htmlFor="password" className="text-violet-400 font-medium text-sm">
                  Password
                </Label>
                <Input
                  id="password"
                  type="password"
                  value={formData.password}
                  onChange={(e) => handleInputChange("password", e.target.value)}
                  className="bg-[#0f0f23] border-gray-600 text-white focus:border-violet-400 focus:ring-violet-400/20 transition-all duration-300 h-10"
                  placeholder="Enter your password"
                  required
                />
              </div>
            </div>

            <Button
              type="submit"
              disabled={loading}
              className="w-full bg-gradient-to-r from-violet-500 to-purple-600 hover:from-violet-600 hover:to-purple-700 text-white py-2.5 text-base font-semibold shadow-lg hover:shadow-violet-500/25 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed mb-3"
            >
              {loading ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Signing In...
                </>
              ) : (
                "Sign In"
              )}
            </Button>
          </form>

          {/* Footer intentionally empty: signup handled via Add User modal */}
        </div>
      </div>
    </div>
  )
}
