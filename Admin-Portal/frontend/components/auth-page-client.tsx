"use client"

import { useState } from "react"
import { useRouter } from "next/navigation"
import { AuthPage } from "@/components/auth-page"
import { setUserSession } from "@/lib/auth"

export function AuthPageClient() {
  const router = useRouter()
  const [mode] = useState<"login">("login")

  const handleLogin = (userData: { id: number; name: string; email: string; role: string; sellerId: number | null}) => {
    try {
      setUserSession(userData)

      // Redirect to dashboard
      router.push("/dashboard")
    } catch (error) {
      console.error("Error setting session:", error)
      
      router.push("/dashboard")
    }
  }

  const handleBack = () => {
    router.push("/")
  }

  const handleSwitchMode = () => {}

  return <AuthPage mode={mode} onLogin={handleLogin} onBack={handleBack} onSwitchMode={handleSwitchMode} />
} 