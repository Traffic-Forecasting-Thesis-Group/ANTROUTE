import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  TextInput,
  ImageBackground,
  Dimensions,
  ScrollView,
  StatusBar,
  KeyboardAvoidingView,
  Platform,
  Image,
  ActivityIndicator,
} from 'react-native';
import { Ionicons, FontAwesome, MaterialCommunityIcons } from '@expo/vector-icons';
import { signUp } from '../api/authService';

const { height, width } = Dimensions.get('window');

export default function SignUpScreen({ navigation }: any) {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isPasswordVisible, setIsPasswordVisible] = useState(false);

  const [focusedInput, setFocusedInput] = useState<string | null>(null);
  const [errors, setErrors] = useState({ name: '', email: '', password: '' });
  const [generalError, setGeneralError] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSignUp = async () => {
    let tempErrors = { name: '', email: '', password: '' };
    let isValid = true;

    // Name Validation
    if (!name.trim()) {
      tempErrors.name = 'Name is required';
      isValid = false;
    }

    // Email Validation with Regex
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!email.trim()) {
      tempErrors.email = 'Email is required';
      isValid = false;
    } else if (!emailRegex.test(email)) {
      tempErrors.email = 'Please enter a valid email address';
      isValid = false;
    }

    // Password Validation
    if (!password.trim()) {
      tempErrors.password = 'Password is required';
      isValid = false;
    } else if (password.length < 6) {
      tempErrors.password = 'Password must be at least 6 characters';
      isValid = false;
    }

    setErrors(tempErrors);
    setGeneralError('');

    if (!isValid) return;

    setIsLoading(true);
    try {
      await signUp(name.trim(), email.trim(), password);
      navigation.replace('Main'); // TODO: swap back to 'Onboarding' once that screen exists
    } catch (error: any) {
      setGeneralError(error?.message || 'Something went wrong. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" translucent backgroundColor="transparent" />
      <ImageBackground
        source={require('../../assets/splash-landing-bg.png')}
        style={styles.background}
        resizeMode="cover"
      >
        <View style={styles.header}>
          <TouchableOpacity onPress={() => navigation.navigate('Landing')} style={styles.backButton}>
            <Ionicons name="arrow-back" size={24} color="white" />
          </TouchableOpacity>
        </View>

        <KeyboardAvoidingView
          behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
          style={{ flex: 1 }}
        >
          <View style={styles.outerBorder}>
            <View style={styles.middleBorder}>
              <View style={styles.whiteCard}>
                <ScrollView
                  showsVerticalScrollIndicator={false}
                  contentContainerStyle={styles.scrollContent}
                  keyboardShouldPersistTaps="handled"
                >
                  <Text style={styles.welcomeTitle}>Get Started</Text>
                  <Text style={styles.subtitle}>Sign up to create an account</Text>

                  {generalError ? (
                    <View style={styles.generalErrorBox}>
                      <Ionicons name="alert-circle" size={16} color="#ef4444" />
                      <Text style={styles.generalErrorText}>{generalError}</Text>
                    </View>
                  ) : null}

                  {/* Name Input */}
                  <View style={styles.inputWrapper}>
                    <View style={[
                      styles.inputContainer,
                      errors.name ? styles.inputErrorBorder : (focusedInput === 'name' ? styles.inputActiveBorder : null)
                    ]}>
                      <MaterialCommunityIcons
                        name="account-outline"
                        size={22}
                        color={errors.name ? "#ef4444" : (focusedInput === 'name' ? "#FFB800" : "#9ca3af")}
                        style={styles.inputIcon}
                      />
                      <TextInput
                        style={styles.input}
                        placeholder="Full Name"
                        placeholderTextColor="#9ca3af"
                        value={name}
                        editable={!isLoading}
                        onFocus={() => setFocusedInput('name')}
                        onBlur={() => setFocusedInput(null)}
                        onChangeText={(text) => { setName(text); setErrors({ ...errors, name: '' }); setGeneralError(''); }}
                      />
                    </View>
                    {errors.name ? <Text style={styles.errorText}>{errors.name}</Text> : null}
                  </View>

                  {/* Email Input */}
                  <View style={styles.inputWrapper}>
                    <View style={[
                      styles.inputContainer,
                      errors.email ? styles.inputErrorBorder : (focusedInput === 'email' ? styles.inputActiveBorder : null)
                    ]}>
                      <Ionicons
                        name="mail-outline"
                        size={20}
                        color={errors.email ? "#ef4444" : (focusedInput === 'email' ? "#FFB800" : "#9ca3af")}
                        style={styles.inputIcon}
                      />
                      <TextInput
                        style={styles.input}
                        placeholder="Email"
                        placeholderTextColor="#9ca3af"
                        value={email}
                        editable={!isLoading}
                        keyboardType="email-address"
                        autoCapitalize="none"
                        autoCorrect={false}
                        onFocus={() => setFocusedInput('email')}
                        onBlur={() => setFocusedInput(null)}
                        onChangeText={(text) => { setEmail(text); setErrors({ ...errors, email: '' }); setGeneralError(''); }}
                      />
                    </View>
                    {errors.email ? <Text style={styles.errorText}>{errors.email}</Text> : null}
                  </View>

                  {/* Password Input */}
                  <View style={styles.inputWrapper}>
                    <View style={[
                      styles.inputContainer,
                      errors.password ? styles.inputErrorBorder : (focusedInput === 'password' ? styles.inputActiveBorder : null)
                    ]}>
                      <Ionicons
                        name="lock-closed-outline"
                        size={20}
                        color={errors.password ? "#ef4444" : (focusedInput === 'password' ? "#FFB800" : "#9ca3af")}
                        style={styles.inputIcon}
                      />
                      <TextInput
                        style={styles.input}
                        placeholder="Password"
                        placeholderTextColor="#9ca3af"
                        value={password}
                        editable={!isLoading}
                        secureTextEntry={!isPasswordVisible}
                        onFocus={() => setFocusedInput('password')}
                        onBlur={() => setFocusedInput(null)}
                        onChangeText={(text) => { setPassword(text); setErrors({ ...errors, password: '' }); setGeneralError(''); }}
                      />
                      <TouchableOpacity onPress={() => setIsPasswordVisible(!isPasswordVisible)} disabled={isLoading}>
                        <Ionicons
                          name={isPasswordVisible ? "eye-outline" : "eye-off-outline"}
                          size={20}
                          color="#9ca3af"
                        />
                      </TouchableOpacity>
                    </View>
                    {errors.password ? <Text style={styles.errorText}>{errors.password}</Text> : null}
                  </View>

                  <TouchableOpacity
                    style={[styles.signUpButton, isLoading && styles.signUpButtonDisabled]}
                    activeOpacity={0.7}
                    onPress={handleSignUp}
                    disabled={isLoading}
                  >
                    {isLoading ? (
                      <ActivityIndicator color="#ffffff" />
                    ) : (
                      <Text style={styles.buttonText}>Sign up</Text>
                    )}
                  </TouchableOpacity>

                  <View style={styles.orContainer}>
                    <View style={styles.line} />
                    <Text style={styles.orText}>or continue with</Text>
                    <View style={styles.line} />
                  </View>

                  <View style={styles.socialRow}>
                    <TouchableOpacity style={[styles.socialBox, styles.shadow]} disabled={isLoading}>
                      <FontAwesome name="facebook" size={28} color="#1877F2" />
                    </TouchableOpacity>
                    <TouchableOpacity style={[styles.socialBox, styles.shadow]} disabled={isLoading}>
                      <Image
                        source={require('../../assets/google-logo-icon.png')}
                        style={{ width: 28, height: 28 }}
                        resizeMode="contain"
                      />
                    </TouchableOpacity>
                  </View>

                  <TouchableOpacity
                    style={styles.footerContainer}
                    onPress={() => navigation.navigate('SignInScreen')}
                    disabled={isLoading}
                  >
                    <Text style={styles.footerText}>Already have an account? <Text style={styles.linkText}>Sign in</Text></Text>
                  </TouchableOpacity>
                </ScrollView>
              </View>
            </View>
          </View>
        </KeyboardAvoidingView>
      </ImageBackground>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#4475F2'
  },
  background: { flex: 1 },
  header: {
    paddingLeft: 20,
    paddingTop: StatusBar.currentHeight ? StatusBar.currentHeight + 10 : 45 },
  backButton: {
    width: 40,
    height: 40
  },
  outerBorder: {
    flex: 1,
    marginTop: height * 0.15,
    backgroundColor: '#FFF4D2',
    borderTopLeftRadius: 50,
    borderTopRightRadius: 50,
    paddingTop: 8 },
  middleBorder: {
    flex: 1,
    backgroundColor: '#D1E0FF',
    borderTopLeftRadius: 45,
    borderTopRightRadius: 45,
    paddingTop: 8 },
  whiteCard: {
    flex: 1,
    backgroundColor: '#ffffff',
    borderTopLeftRadius: 40,
    borderTopRightRadius: 40,
    overflow: 'hidden' },
  scrollContent: {
    alignItems: 'center',
    paddingHorizontal: 35,
    paddingTop: 30,
    paddingBottom: 40
  },
  welcomeTitle: {
    fontSize: 34,
    fontWeight: 'bold',
    color: '#FFB800',
    marginBottom: 5
  },
  subtitle: {
    fontSize: 15,
    color: '#6b7280',
    marginBottom: 18,
    textAlign: 'center'
  },
  generalErrorBox: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#fff5f5',
    borderWidth: 1,
    borderColor: '#fecaca',
    borderRadius: 12,
    paddingVertical: 10,
    paddingHorizontal: 14,
    width: '100%',
    marginBottom: 15,
    gap: 8,
  },
  generalErrorText: {
    color: '#ef4444',
    fontSize: 13,
    fontWeight: '600',
    flexShrink: 1,
  },
  inputWrapper: {
    width: '100%',
    marginBottom: 15
  },
  inputContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#F8F9FA',
    borderRadius: 18,
    paddingHorizontal: 20,
    width: '100%',
    height: 60,
    borderWidth: 1.5,
    borderColor: 'transparent'
  },
  inputActiveBorder: {
    borderColor: '#FFB800',
    backgroundColor: '#fff'
  },
  inputErrorBorder: {
    borderColor: '#ef4444',
    backgroundColor: '#fff5f5'
  },
  errorText: {
    color: '#ef4444',
    fontSize: 11,
    marginTop: 4,
    marginLeft: 10,
    fontWeight: '600'
  },
  inputIcon: { marginRight: 15 },
  input: {
    flex: 1,
    fontSize: 16,
    color: '#374151',
    fontWeight: '500'
  },
  signUpButton: {
    backgroundColor: '#FFB800',
    width: '100%',
    height: 55,
    borderRadius: 30,
    justifyContent: 'center',
    alignItems: 'center',
    marginTop: 10,
    marginBottom: 20,
    elevation: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.2,
    shadowRadius: 5,
  },
  signUpButtonDisabled: {
    opacity: 0.7,
  },
  buttonText: {
    color: '#ffffff',
    fontSize: 18,
    fontWeight: 'bold'
  },
  orContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 20,
    width: '100%'
  },
  line: {
    flex: 1,
    height: 1,
    backgroundColor: '#E5E7EB'
  },
  orText: {
    marginHorizontal: 15,
    fontSize: 14,
    color: '#9ca3af',
    fontWeight: '500'
  },
  socialRow: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 20,
    marginBottom: 20
  },
  socialBox: {
    width: width * 0.22,
    height: 55,
    borderWidth: 1,
    borderColor: '#F3F4F6',
    borderRadius: 18,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#fff'
  },
  shadow: {
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 3,
  },
  footerContainer: { marginTop: 10 },
  footerText: {
    fontSize: 14,
    color: '#6b7280'
  },
  linkText: {
    color: '#FFB800',
    fontWeight: '800'
  },
});
