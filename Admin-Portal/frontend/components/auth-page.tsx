"use client";

import type React from "react";
import { useState } from "react";
import { ArrowLeft, MessageCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { saveAuth } from "@/lib/authUtils";
import router from "next/router";

interface AuthPageProps {
  mode: "login" | "register";
  onLogin: (userData: { businessName: string; email: string }) => void;
  onBack: () => void;
  onSwitchMode: (mode: "login" | "register") => void;
}

export function AuthPage({
  mode,
  onLogin,
  onBack,
  onSwitchMode,
}: AuthPageProps) {
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    phone: "",
    address: "",
    password: "",
    confirmPassword: "",
    role: "seller"
  });

  const [passwordsMatch, setPasswordsMatch] = useState(true);
  const [showPasswordError, setShowPasswordError] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    try {
      if (mode === "register") {
        const res = await fetch("http://localhost:7001/api/auth/register", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(formData),
        });

        const data = await res.json();

        if (!res.ok) {
          setError(data.error || "Registration failed");
          return;
        }
        onLogin({
          businessName: formData.name || "Demo Business",
          email: formData.email,
        });
      } else {
        const password = formData.password;
        const email = formData.email;

        const res = await fetch("http://localhost:7001/api/auth/login", {
          method: "POST",
          credentials: "include",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email, password }),
        });

        const data = await res.json();

        if (!res.ok) {
          setError( data.message || 'Invalid credentials');
          return;
        }

        if (data.user.role === 'admin') {
          router.push('/admin/dashboard');
        } else if (data.user.role === 'seller') {
          onLogin({
            businessName: data.user.name || "Demo Business",
            email: formData.email,
          });
        } else {
          router.push('/'); // fallback
        }
      }
    } catch (error) {
      setError("Something went wrong");
    }
  };

  const handleInputChange = (field: string, value: string) => {
    setFormData((prev) => {
      const newData = { ...prev, [field]: value };

      // Validate passwords in real-time
      if (field === "password" || field === "confirmPassword") {
        const password = field === "password" ? value : prev.password;
        const confirmPassword =
          field === "confirmPassword" ? value : prev.confirmPassword;
        validatePasswords(password, confirmPassword);
      }

      return newData;
    });
  };

  const validatePasswords = (password: string, confirmPassword: string) => {
    if (confirmPassword.length > 0) {
      const match = password === confirmPassword;
      setPasswordsMatch(match);
      setShowPasswordError(!match && confirmPassword.length > 0);
    } else {
      setShowPasswordError(false);
      setPasswordsMatch(true);
    }
  };

  return (
    <div className="h-screen bg-gradient-to-br from-[#0f0f23] to-[#1a1a2e] flex items-center justify-center px-6 overflow-hidden">
      <div
        className={`w-full ${mode === "register" ? "max-w-4xl" : "max-w-md"}`}
      >
        {/* Back button */}
        <Button
          variant="ghost"
          onClick={onBack}
          className="mb-2 text-violet-400 hover:text-emerald-400 hover:bg-transparent transition-colors duration-300"
        >
          <ArrowLeft className="w-4 h-4 mr-2" />
          Back to Home
        </Button>

        {/* error messages */}
        {error && <p className="text-xs mb-4 text-center" style={{ color: 'red' }}>{error}</p>}

        {/* Form container */}
        <div className="bg-gradient-to-b from-[#1a1a2e] to-[#16213e] p-6 rounded-xl border border-gray-700 shadow-2xl">
          {/* Header */}
          <div className="text-center mb-8">
            <MessageCircle className="w-8 h-8 text-violet-400 mx-auto mb-2" />
            <h2 className="text-xl font-bold mb-1">
              {mode === "login" ? "Welcome Back" : "Create Account"}
            </h2>
            <p className="text-gray-400 text-sm">
              {mode === "login"
                ? "Sign in to your WhatsApp Business dashboard"
                : "Start automating your WhatsApp Business today"}
            </p>
          </div>

          <form onSubmit={handleSubmit}>
            {mode === "register" ? (
              <>
                {/* Two-column layout for registration */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4">
                  {/* Left Column */}
                  <div className="space-y-5">
                    <div className="space-y-3">
                      <Label
                        htmlFor="businessName"
                        className="text-violet-400 font-medium text-sm"
                      >
                        Business Name
                      </Label>
                      <Input
                        id="name"
                        type="text"
                        value={formData.name}
                        onChange={(e) =>
                          handleInputChange("name", e.target.value)
                        }
                        className="bg-[#0f0f23] border-gray-600 text-white focus:border-violet-400 focus:ring-violet-400/20 transition-all duration-300 h-10"
                        placeholder="Enter your business name"
                        required
                      />
                    </div>

                    <div className="space-y-3">
                      <Label
                        htmlFor="email"
                        className="text-violet-400 font-medium text-sm"
                      >
                        Email Address
                      </Label>
                      <Input
                        id="email"
                        type="email"
                        value={formData.email}
                        onChange={(e) =>
                          handleInputChange("email", e.target.value)
                        }
                        className="bg-[#0f0f23] border-gray-600 text-white focus:border-violet-400 focus:ring-violet-400/20 transition-all duration-300 h-10"
                        placeholder="Enter your email"
                        required
                      />
                    </div>

                    <div className="space-y-3 ">
                      <Label
                        htmlFor="password"
                        className="text-violet-400 font-medium text-sm"
                      >
                        Password
                      </Label>
                      <Input
                        id="password"
                        type="password"
                        value={formData.password}
                        onChange={(e) =>
                          handleInputChange("password", e.target.value)
                        }
                        className="bg-[#0f0f23] border-gray-600 text-white focus:border-violet-400 focus:ring-violet-400/20 transition-all duration-300 h-10"
                        placeholder="Enter your password"
                        required
                      />
                    </div>
                  </div>

                  {/* Right Column */}
                  <div className="space-y-5">
                    <div className="space-y-3">
                      <Label
                        htmlFor="whatsappNumber"
                        className="text-violet-400 font-medium text-sm"
                      >
                        WhatsApp Number
                      </Label>
                      <Input
                        id="phone"
                        type="tel"
                        value={formData.phone}
                        onChange={(e) =>
                          handleInputChange("phone", e.target.value)
                        }
                        className="bg-[#0f0f23] border-gray-600 text-white focus:border-violet-400 focus:ring-violet-400/20 transition-all duration-300 h-10"
                        placeholder="+1 (555) 123-4567"
                        required
                      />
                    </div>

                    <div className="space-y-3">
                      <Label
                        htmlFor="address"
                        className="text-violet-400 font-medium text-sm"
                      >
                        Business Address
                      </Label>
                      <Input
                        id="address"
                        type="text"
                        value={formData.address}
                        onChange={(e) =>
                          handleInputChange("address", e.target.value)
                        }
                        className="bg-[#0f0f23] border-gray-600 text-white focus:border-violet-400 focus:ring-violet-400/20 transition-all duration-300 h-10"
                        placeholder="Enter your business address"
                        required
                      />
                    </div>

                    <div className="space-y-3">
                      <Label
                        htmlFor="confirmPassword"
                        className="text-violet-400 font-medium text-sm"
                      >
                        Confirm Password
                      </Label>
                      <Input
                        id="confirmPassword"
                        type="password"
                        value={formData.confirmPassword}
                        onChange={(e) =>
                          handleInputChange("confirmPassword", e.target.value)
                        }
                        className={`bg-[#0f0f23] border-gray-600 text-white focus:ring-violet-400/20 transition-all duration-300 h-10 ${
                          showPasswordError
                            ? "border-red-500 focus:border-red-500"
                            : passwordsMatch &&
                              formData.confirmPassword.length > 0
                            ? "border-emerald-500 focus:border-emerald-500"
                            : "focus:border-violet-400"
                        }`}
                        placeholder="Confirm your password"
                        required
                      />
                      {formData.confirmPassword.length > 0 && (
                        <div className="flex items-center space-x-2 text-xs">
                          {passwordsMatch ? (
                            <>
                              <div className="w-1.5 h-1.5 bg-emerald-400 rounded-full animate-pulse"></div>
                              <span className="text-emerald-400 font-medium">
                                Passwords match
                              </span>
                            </>
                          ) : (
                            <>
                              <div className="w-1.5 h-1.5 bg-red-400 rounded-full animate-pulse"></div>
                              <span className="text-red-400 font-medium">
                                Passwords do not match
                              </span>
                            </>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </>
            ) : (
              <>
                {/* Single column layout for login */}
                <div className="space-y-3 mb-4">
                  <div className="space-y-1">
                    <Label
                      htmlFor="email"
                      className="text-violet-400 font-medium text-sm"
                    >
                      Email Address
                    </Label>
                    <Input
                      id="email"
                      type="email"
                      value={formData.email}
                      onChange={(e) =>
                        handleInputChange("email", e.target.value)
                      }
                      className="bg-[#0f0f23] border-gray-600 text-white focus:border-violet-400 focus:ring-violet-400/20 transition-all duration-300 h-10"
                      placeholder="Enter your email"
                      required
                    />
                  </div>

                  <div className="space-y-1">
                    <Label
                      htmlFor="password"
                      className="text-violet-400 font-medium text-sm"
                    >
                      Password
                    </Label>
                    <Input
                      id="password"
                      type="password"
                      value={formData.password}
                      onChange={(e) =>
                        handleInputChange("password", e.target.value)
                      }
                      className="bg-[#0f0f23] border-gray-600 text-white focus:border-violet-400 focus:ring-violet-400/20 transition-all duration-300 h-10"
                      placeholder="Enter your password"
                      required
                    />
                  </div>
                </div>
              </>
            )}

            <Button
              type="submit"
              disabled={
                mode === "register" &&
                (!passwordsMatch || formData.confirmPassword.length === 0)
              }
              className="w-full bg-gradient-to-r from-violet-500 to-purple-600 hover:from-violet-600 hover:to-purple-700 text-white py-2.5 text-base font-semibold shadow-lg hover:shadow-violet-500/25 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed mb-4 mt-8"
            >
              {mode === "login" ? "Sign In" : "Create Account"}
            </Button>
          </form>

          {/* Footer */}
          <div className="text-center">
            <p className="text-gray-400 text-sm">
              {mode === "login"
                ? "Don't have an account? "
                : "Already have an account? "}
              <button
                onClick={() =>
                  onSwitchMode(mode === "login" ? "register" : "login")
                }
                className="text-violet-400 hover:text-emerald-400 transition-colors duration-300 font-semibold"
              >
                {mode === "login" ? "Sign up" : "Sign in"}
              </button>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
