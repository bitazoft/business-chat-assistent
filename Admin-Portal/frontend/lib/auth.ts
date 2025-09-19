export interface User {
  id: number; name: string; email: string; role: string; sellerId: number | null;
}

// Client-side only authentication functions
export function getCurrentUserClient(): User | null {
  if (typeof window === "undefined") return null

  try {
    const savedUser = localStorage.getItem("whatsapp_business_user")
    const sessionExpiry = localStorage.getItem("whatsapp_business_session_expiry")

    if (!savedUser || !sessionExpiry) {
      return null
    }

    const now = new Date().getTime()
    const expiry = Number.parseInt(sessionExpiry)

    if (isNaN(expiry) || now >= expiry) {
      // Session expired, clear storage
      localStorage.removeItem("whatsapp_business_user")
      localStorage.removeItem("whatsapp_business_session_expiry")
      return null
    }

    const userData = JSON.parse(savedUser)

    if (!userData.email || !userData.name || !userData.role) {
      console.error("Invalid user data structure:", userData)
      // clearUserSession()
      return null
    }

    return userData
  } catch (error) {
    console.error("Error getting current user from localStorage:", error)
    // Clear corrupted data
    clearUserSession()
    return null
  }
}

export function setUserSession(user: User) {
  if (typeof window === "undefined") return

  const expiryTime = new Date().getTime() + 24 * 60 * 60 * 1000 // 24 hours

  try {
    // Save to localStorage
    localStorage.setItem("whatsapp_business_user", JSON.stringify(user))
    localStorage.setItem("whatsapp_business_session_expiry", expiryTime.toString())

    // Also set cookies for middleware
    const encodedUser = encodeURIComponent(JSON.stringify(user))
    document.cookie = `whatsapp_business_user=${encodedUser}; path=/; max-age=${24 * 60 * 60}; SameSite=Lax`
    document.cookie = `whatsapp_business_session_expiry=${expiryTime}; path=/; max-age=${24 * 60 * 60}; SameSite=Lax`
  } catch (error) {
    console.error("Error setting user session:", error)
  }
}

export function clearUserSession() {
  if (typeof window === "undefined") return

  try {
    // Clear localStorage
    localStorage.removeItem("whatsapp_business_user")
    localStorage.removeItem("whatsapp_business_session_expiry")

    // Clear cookies
    document.cookie = "whatsapp_business_user=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT; SameSite=Lax"
    document.cookie = "whatsapp_business_session_expiry=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT; SameSite=Lax"
  } catch (error) {
    console.error("Error clearing user session:", error)
  }
}

// Cookie helper functions for client-side use
export function getCookieValue(name: string): string | null {
  if (typeof window === "undefined") return null

  const value = `; ${document.cookie}`
  const parts = value.split(`; ${name}=`)
  if (parts.length === 2) {
    const cookieValue = parts.pop()?.split(";").shift()
    return cookieValue ? decodeURIComponent(cookieValue) : null
  }
  return null
}

export function getUserFromCookies(): User | null {
  if (typeof window === "undefined") return null

  try {
    const userCookie = getCookieValue("whatsapp_business_user")
    const sessionExpiry = getCookieValue("whatsapp_business_session_expiry")

    if (!userCookie || !sessionExpiry) {
      return null
    }

    const now = new Date().getTime()
    const expiry = Number.parseInt(sessionExpiry)

    if (isNaN(expiry) || now >= expiry) {
      return null
    }

    const userData = JSON.parse(userCookie)

    if (!userData.email || !userData.name || !userData.role) {
      return null
    }

    return userData
  } catch (error) {
    console.error("Error getting user from cookies:", error)
    return null
  }
}

// Unified function that checks both localStorage and cookies
export function getCurrentUser(): User | null {
  // Try localStorage first
  let user = getCurrentUserClient()

  // If not found, try cookies
  if (!user) {
    user = getUserFromCookies()

    // If found in cookies, sync to localStorage
    if (user) {
      const sessionExpiry = getCookieValue("whatsapp_business_session_expiry")
      if (sessionExpiry) {
        localStorage.setItem("whatsapp_business_user", JSON.stringify(user))
        localStorage.setItem("whatsapp_business_session_expiry", sessionExpiry)
      }
    }
  }

  return user
}

export function isAdmin(user: User | null): boolean {
  return user?.role === "admin"
}

export function hasRole(user: User | null, role: "admin" | "user"): boolean {
  return user?.role === role
}