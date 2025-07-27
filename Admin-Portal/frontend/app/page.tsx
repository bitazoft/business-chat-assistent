"use client"

import { useState, useEffect } from "react"
import { LandingPage } from "@/components/landing-page"
import { AuthPage } from "@/components/auth-page"
import { Dashboard } from "@/components/dashboard"
import { useRouter } from "next/navigation"

export default function HomePage() {
  const router = useRouter()
  const [currentPage, setCurrentPage] = useState<"landing" | "auth" | "dashboard">("landing")
  const [authMode, setAuthMode] = useState<"login" | "register">("login")
  const [user, setUser] = useState<{ id: number; name: string; email: string; role: string} | null>(null)
  const [loading, setLoading] = useState(true)

  // Check for existing session on app load
  useEffect(() => {
    checkUserSession()
  }, [])

  const checkUserSession = async () => {
    try {
      const res = await fetch("http://localhost:7001/api/auth/me", {
        method: "GET",
        credentials: "include",
      });
  
      if (res.ok) {
        const data = await res.json();
        setUser(data.user);
        router.push("/dashboard")
      } else {
        console.log("Not logged in");
        setLoading(false)
      }
    } catch (error) {
      console.error("Session check failed:", error)
    } finally {
      setLoading(false)
    }
  }

  const handleLogin = (userData: { id: number; name: string; email: string; role: string}) => {
    // Set session expiry to 24 hours from now
    const expiryTime = new Date().getTime() + 24 * 60 * 60 * 1000

    // Save user data and session
    localStorage.setItem("whatsapp_business_user", JSON.stringify(userData))
    localStorage.setItem("whatsapp_business_session_expiry", expiryTime.toString())

    setUser(userData)
    // Redirect to dashboard after successful login
    setCurrentPage("dashboard")
  }

  const handleAuth = (mode: "login" | "register") => {
    setAuthMode(mode)
    setCurrentPage("auth")
  }

  const handleLogout = async () => {
    try {
      await fetch("http://localhost:7001/api/auth/logout", {
        method: "POST",
        credentials: "include",
      });
      setCurrentPage("landing")
    } catch (error) {
      console.error("Logout failed", error);
    }
}

  // Show loading spinner while checking session
  if (loading) {
    return (
      <div className="min-h-screen bg-[#0f0f23] flex items-center justify-center">
        <div className="flex items-center space-x-2">
          <div className="w-4 h-4 bg-violet-400 rounded-full animate-pulse"></div>
          <div className="text-violet-400 text-xl">Loading...</div>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-[#0f0f23] text-white font-inter">
      {currentPage === "landing" && <LandingPage />}
      {currentPage === "auth" && (
        <AuthPage
          mode={authMode}
          onLogin={handleLogin}
          onBack={() => setCurrentPage("landing")}
          onSwitchMode={(mode) => setAuthMode(mode)}
        />
      )}
      {currentPage === "dashboard" && user && (
        <Dashboard user={user} onLogout={handleLogout} />
      )}
    </div>
  )
}
