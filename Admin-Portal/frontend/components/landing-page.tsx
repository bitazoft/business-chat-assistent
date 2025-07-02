"use client"

import { useState, useEffect } from "react"
import { MessageCircle, Zap, TrendingUp, Users } from "lucide-react"
import { Button } from "@/components/ui/button"

interface LandingPageProps {
  onAuth: (mode: "login" | "register") => void
}

export function LandingPage({ onAuth }: LandingPageProps) {
  const [isVisible, setIsVisible] = useState(false)

  useEffect(() => {
    setIsVisible(true)
  }, [])

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#0f0f23] to-[#1a1a2e] relative overflow-hidden">
      {/* Animated background elements */}
      <div className="absolute inset-0">
        <div className="absolute top-20 left-10 w-2 h-2 bg-violet-400 rounded-full animate-pulse"></div>
        <div className="absolute top-40 right-20 w-1 h-1 bg-emerald-400 rounded-full animate-ping"></div>
        <div className="absolute bottom-32 left-1/4 w-1.5 h-1.5 bg-violet-400 rounded-full animate-pulse delay-1000"></div>
        <div className="absolute bottom-20 right-1/3 w-1 h-1 bg-emerald-400 rounded-full animate-ping delay-500"></div>
      </div>

      <div className="relative z-10 container mx-auto px-6 py-8">
        {/* Header */}
        <header className="flex justify-between items-center mb-16">
          <div className="flex items-center space-x-2">
            <MessageCircle className="w-8 h-8 text-violet-400" />
            <span className="text-xl font-bold">WhatsApp Business AI</span>
          </div>
          <div className="space-x-4">
            <Button
              variant="outline"
              onClick={() => onAuth("login")}
              className="border-violet-400 text-violet-400 hover:bg-violet-400 hover:text-white transition-all duration-300"
            >
              Login
            </Button>
            <Button
              onClick={() => onAuth("register")}
              className="bg-gradient-to-r from-violet-500 to-purple-600 hover:from-violet-600 hover:to-purple-700 text-white transition-all duration-300 shadow-lg hover:shadow-violet-500/25"
            >
              Register
            </Button>
          </div>
        </header>

        {/* Hero Section */}
        <div className="text-center mb-20">
          <div
            className={`transition-all duration-1000 ${isVisible ? "translate-y-0 opacity-100" : "translate-y-10 opacity-0"}`}
          >
            <h1 className="text-6xl md:text-7xl font-bold mb-6 leading-tight">
              <span className="text-white">AI Chat Assistant</span>
              <br />
              <span className="bg-gradient-to-r from-violet-400 to-purple-600 bg-clip-text text-transparent">
                for WhatsApp Business
              </span>
            </h1>
            <p className="text-xl text-gray-300 mb-8 max-w-2xl mx-auto">
              Automate customer support, manage orders, and boost sales with our intelligent WhatsApp Business solution
            </p>
            <div className="space-x-4">
              <Button
                size="lg"
                onClick={() => onAuth("register")}
                className="bg-gradient-to-r from-violet-500 to-purple-600 hover:from-violet-600 hover:to-purple-700 text-white px-8 py-4 text-lg font-semibold shadow-lg hover:shadow-violet-500/25 transition-all duration-300"
              >
                Get Started Free
              </Button>
              <Button
                size="lg"
                variant="outline"
                className="border-emerald-400 text-emerald-400 hover:bg-emerald-400 hover:text-white transition-all duration-300 px-8 py-4 text-lg bg-transparent"
              >
                Watch Demo
              </Button>
            </div>
          </div>
        </div>

        {/* Features */}
        <div
          className={`grid md:grid-cols-3 gap-8 mb-20 transition-all duration-1000 delay-500 ${isVisible ? "translate-y-0 opacity-100" : "translate-y-10 opacity-0"}`}
        >
          <div className="text-center p-6 rounded-lg border border-gray-700 hover:border-violet-400 transition-all duration-300 hover:shadow-lg hover:shadow-violet-500/10 bg-gradient-to-b from-[#1a1a2e] to-[#16213e]">
            <Zap className="w-12 h-12 text-emerald-400 mx-auto mb-4" />
            <h3 className="text-xl font-semibold mb-2">Instant Automation</h3>
            <p className="text-gray-400">Respond to customers 24/7 with AI-powered chat automation</p>
          </div>
          <div className="text-center p-6 rounded-lg border border-gray-700 hover:border-violet-400 transition-all duration-300 hover:shadow-lg hover:shadow-violet-500/10 bg-gradient-to-b from-[#1a1a2e] to-[#16213e]">
            <TrendingUp className="w-12 h-12 text-emerald-400 mx-auto mb-4" />
            <h3 className="text-xl font-semibold mb-2">Sales Analytics</h3>
            <p className="text-gray-400">Track performance and optimize your business strategy</p>
          </div>
          <div className="text-center p-6 rounded-lg border border-gray-700 hover:border-violet-400 transition-all duration-300 hover:shadow-lg hover:shadow-violet-500/10 bg-gradient-to-b from-[#1a1a2e] to-[#16213e]">
            <Users className="w-12 h-12 text-emerald-400 mx-auto mb-4" />
            <h3 className="text-xl font-semibold mb-2">Customer Management</h3>
            <p className="text-gray-400">Organize and manage all your customer interactions</p>
          </div>
        </div>
      </div>
    </div>
  )
}
