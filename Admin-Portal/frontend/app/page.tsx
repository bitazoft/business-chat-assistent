"use client"

import { useState } from "react"
import { LandingPage } from "@/components/landing-page"
import { AuthPage } from "@/components/auth-page"
import { Dashboard } from "@/components/dashboard"

export default function App() {
  const [currentPage, setCurrentPage] = useState<"landing" | "auth" | "dashboard">("landing")
  const [authMode, setAuthMode] = useState<"login" | "register">("login")
  const [user, setUser] = useState<{ businessName: string; email: string } | null>(null)

  const handleAuth = (mode: "login" | "register") => {
    setAuthMode(mode)
    setCurrentPage("auth")
  }

  const handleLogin = (userData: { businessName: string; email: string }) => {
    setUser(userData)
    setCurrentPage("dashboard")
  }

  const handleLogout = () => {
    setUser(null)
    setCurrentPage("landing")
  }

  return (
    <div className="min-h-screen bg-[#0f0f23] text-white font-inter">
      {currentPage === "landing" && <LandingPage onAuth={handleAuth} />}
      {currentPage === "auth" && (
        <AuthPage
          mode={authMode}
          onLogin={handleLogin}
          onBack={() => setCurrentPage("landing")}
          onSwitchMode={(mode) => setAuthMode(mode)}
        />
      )}
      {currentPage === "dashboard" && user && <Dashboard user={user} onLogout={handleLogout} />}
    </div>
  )
}
