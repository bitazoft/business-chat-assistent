import { type NextRequest, NextResponse } from "next/server"

export async function POST(request: NextRequest) {
  try {
    const { businessName, email, password, whatsappNumber, address } = await request.json()

    // Mock validation
    if (!businessName || !email || !password || !whatsappNumber || !address) {
      return NextResponse.json({ success: false, message: "All fields are required" }, { status: 400 })
    }

    if (password.length < 6) {
      return NextResponse.json({ success: false, message: "Password must be at least 6 characters" }, { status: 400 })
    }

    // Mock successful registration
    const user = {
      businessName,
      email,
      whatsappNumber,
      address,
    }

    return NextResponse.json({
      success: true,
      user,
      message: "Registration successful",
    })
  } catch (error) {
    return NextResponse.json({ success: false, message: "Internal server error" }, { status: 500 })
  }
}
