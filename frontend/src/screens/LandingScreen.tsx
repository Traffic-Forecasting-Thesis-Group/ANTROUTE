import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ImageBackground,
  SafeAreaView,
  Dimensions,
  Image, 
} from 'react-native';

const { height } = Dimensions.get('window');

export default function LandingScreen({ navigation }: any) {
  return (
    <View style={styles.container}>
      <ImageBackground
        source={require('../../assets/splash-landing-bg.png')}
        style={styles.background}
        resizeMode="cover"
      >
        <View style={styles.outerBorder}>
          <View style={styles.middleBorder}>
            <View style={styles.whiteCard}>
              <SafeAreaView style={styles.innerContent}>
                
                {/* Logo Section */}
                <Image 
                  source={require('../../assets/logo-no-bg.png')}
                  style={styles.logoImage}
                  resizeMode="contain"
                />

                <Text style={styles.appName}>
                  <Text style={styles.blueText}>ANT</Text>
                  <Text style={styles.yellowText}>Route</Text>
                </Text>

                <Text style={styles.description}>
                  Optimized routing for Metro Manila's daily traffic
                </Text>

                <TouchableOpacity
                  style={styles.signUpButton}
                  activeOpacity={0.8}
                  onPress={() => navigation.navigate('SignUpScreen')}
                >
                  <Text style={styles.buttonText}>Sign Up</Text>
                </TouchableOpacity>

                <View style={styles.orContainer}>
                  <View style={styles.line} />
                  <Text style={styles.orText}>OR</Text>
                  <View style={styles.line} />
                </View>

                <TouchableOpacity
                  style={styles.signInButton}
                  activeOpacity={0.8}
                  onPress={() => navigation.navigate('SignInScreen')}
                >
                  <Text style={styles.buttonText}>Sign In</Text>
                </TouchableOpacity>
              </SafeAreaView>
            </View>
          </View>
        </View>
      </ImageBackground>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  background: { flex: 1 },
  
  outerBorder: {
    flex: 1,
    marginTop: height * 0.48,
    backgroundColor: '#FFF4D2', 
    borderTopLeftRadius: 50,
    borderTopRightRadius: 50,
    paddingTop: 8 
  },
  middleBorder: {
    flex: 1,
    backgroundColor: '#D1E0FF', 
    borderTopLeftRadius: 45,
    borderTopRightRadius: 45,
    paddingTop: 8, 
  },

  whiteCard: {
    flex: 1,
    backgroundColor: '#ffffff',
    borderTopLeftRadius: 40,
    borderTopRightRadius: 40,
    paddingHorizontal: 25,
    paddingTop: 20,
    alignItems: 'center',
  },

  innerContent: {
    alignItems: 'center',
    width: '100%',
    paddingTop: 25, 
  },

  logoImage: {
    width: 90, 
    height: 90,
    marginBottom: 10,
  },

  appName: {
    fontSize: 34,
  fontWeight: '800',
  letterSpacing: 0.5,
  marginBottom: 10,
  },
  blueText: { color: '#4475F2' },
  yellowText: { color: '#FFB800' },

  description: {
    textAlign: 'center',
    color: '#6b7280',
    fontSize: 12,
    lineHeight: 18,
    marginBottom: 30,
    paddingHorizontal: 20,
  },

  signUpButton: {
    backgroundColor: '#FFB800',
    width: '80%',
    paddingVertical: 14,
    borderRadius: 25,
    alignItems: 'center',
    marginBottom: 10,
  },

  signInButton: {
    backgroundColor: '#4475F2',
    width: '80%',
    paddingVertical: 14,
    borderRadius: 25,
    alignItems: 'center',
  },

  buttonText: { color: '#ffffff', fontSize: 16, fontWeight: 'bold' },

  orContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginVertical: 12,
    width: '80%',
  },
  line: { 
    flex: 1, 
    height: 1, 
    backgroundColor: '#e5e7eb' 
  },
  orText: { 
    
    marginHorizontal: 12, 
    fontSize: 12, 
    fontWeight: 'bold', 
    color: '#9ca3af' 
  },
});