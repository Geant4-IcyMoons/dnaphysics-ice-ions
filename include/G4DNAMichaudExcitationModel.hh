//
// ********************************************************************
// // * License and Disclaimer                                           *
// // *                                                                  *
// // * The  Geant4 software  is  copyright of the Copyright Holders  of *
// // * the Geant4 Collaboration.  It is provided  under  the terms  and *
// // * conditions of the Geant4 Software License,  included in the file *
// // * LICENSE and available at  http://cern.ch/geant4/license .  These *
// // * include a list of copyright holders.                             *
// // *                                                                  *
// // * Neither the authors of this software system, nor their employing *
// // * institutes,nor the agencies providing financial support for this *
// // * work  make  any representation or  warranty, express or implied, *
// // * regarding  this  software system or assume any liability for its *
// // * use.  Please see the license in the file  LICENSE  and URL above *
// // * for the full disclaimer and the limitation of liability.         *
// // *                                                                  *
// // * This  code  implementation is the result of  the  scientific and *
// // * technical work of the GEANT4 collaboration.                      *
// // * By using,  copying,  modifying or  distributing the software (or *
// // * any work based  on the software)  you  agree  to acknowledge its *
// // * use  in  resulting  scientific  publications,  and indicate your *
// // * acceptance of all terms of the Geant4 Software license.          *
// // ********************************************************************
// //
// //

// //....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

// // Created by G. Yoffe

// #ifndef G4DNAMichaudExcitationModel_h
// #define G4DNAMichaudExcitationModel_h 1

// #include <deque>
// #include <CLHEP/Units/SystemOfUnits.h>

// #include "G4VEmModel.hh"
// #include "G4ParticleChangeForGamma.hh"
// #include "G4Electron.hh"
// #include "G4NistManager.hh"

// class G4DNAMichaudExcitationModel : public G4VEmModel
// {
// public:
//   G4DNAMichaudExcitationModel(const G4ParticleDefinition* p = nullptr,
//                              const G4String& nam = "DNAMichaudExcitationModel");

//   ~G4DNAMichaudExcitationModel() override;

//   G4DNAMichaudExcitationModel & operator=(const G4DNAMichaudExcitationModel &right) = delete;
//   G4DNAMichaudExcitationModel(const G4DNAMichaudExcitationModel&) = delete;

//   void Initialise(const G4ParticleDefinition*,
//                           const G4DataVector&) override;

//   G4double CrossSectionPerVolume(const G4Material* material,
//                                          const G4ParticleDefinition* p,
//                                          G4double ekin,
//                                          G4double emin,
//                                          G4double emax) override;

//   void SampleSecondaries(std::vector<G4DynamicParticle*>*,
//                                  const G4MaterialCutsCouple*,
//                                  const G4DynamicParticle*,
//                                  G4double tmin,
//                                  G4double maxEnergy) override;

//   // Cross section

//   G4double PartialCrossSection(G4double energy, G4int level);
//   G4double TotalCrossSection(G4double t);

//   inline void ExtendLowEnergyLimit(G4double /*threshold*/);

//   inline void SetVerboseLevel(G4int verbose)
//   {
//     verboseLevel = verbose;
//   }

//   inline void SelectStationary(G4bool input); 

// protected:

//   G4ParticleChangeForGamma* fParticleChangeForGamma;

// private:

//   G4bool statCode;

//   // Water density table
//   const std::vector<G4double>* fpWaterDensity;

//   G4bool isInitialised{false};
//   G4int verboseLevel;

//   // Cross section

//   G4int RandomSelect(G4double energy);
//   G4int nLevels;
//   G4double VibrationEnergy(G4int level);
//   G4double Sum(G4double k);
//   G4double LinInterpolate(G4double e1,
//                           G4double e2,
//                           G4double e,
//                           G4double xs1,
//                           G4double xs2);

//   //
// //  typedef std::map<double, std::map<double, double> > TriDimensionMap;
// //  TriDimensionMap map1;
//   std::vector<G4double> tdummyVec;
//   std::vector<std::vector<G4double>> fEnergyLevelXS;
//   std::vector<G4double> fEnergyTotalXS;

// };

// //....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

// inline void G4DNAMichaudExcitationModel::ExtendLowEnergyLimit(G4double threshold)
// {
//   if(threshold < 2 * CLHEP::eV)
//     G4Exception("*** WARNING : the G4DNAMichaudExcitationModel class is not "
//                 "validated below 2 eV !",
//                 "", JustWarning, "");
  
//   SetLowEnergyLimit(threshold);
// }

// //....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

// inline void G4DNAMichaudExcitationModel::SelectStationary (G4bool input)
// { 
//     statCode = input; 
// }		 

// //....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

// #endif
//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
//
// Created by G. Yoffe
//

#ifndef G4DNAMichaudExcitationModel_h
#define G4DNAMichaudExcitationModel_h 1

#include <deque>
#include <vector>
#include <array>
#include <cmath>

#include <CLHEP/Units/SystemOfUnits.h>

#include "G4VEmModel.hh"
#include "G4ParticleChangeForGamma.hh"
#include "G4Electron.hh"
#include "G4NistManager.hh"

class G4DNAMichaudExcitationModel : public G4VEmModel
{
 public:
  G4DNAMichaudExcitationModel(const G4ParticleDefinition* p = nullptr,
                              const G4String& nam = "DNAMichaudExcitationModel");

  ~G4DNAMichaudExcitationModel() override;

  G4DNAMichaudExcitationModel& operator=(const G4DNAMichaudExcitationModel&) = delete;
  G4DNAMichaudExcitationModel(const G4DNAMichaudExcitationModel&) = delete;

  // Standard G4VEmModel hooks
  void Initialise(const G4ParticleDefinition*, const G4DataVector&) override;

  G4double CrossSectionPerVolume(const G4Material* material,
                                 const G4ParticleDefinition* p,
                                 G4double ekin,
                                 G4double emin,
                                 G4double emax) override;

  void SampleSecondaries(std::vector<G4DynamicParticle*>*,
                         const G4MaterialCutsCouple*,
                         const G4DynamicParticle*,
                         G4double tmin,
                         G4double maxEnergy) override;

  // XS accessors
  G4double PartialCrossSection(G4double energy, G4int level);
  G4double PartialCrossSection(G4double energy, G4int level) const;
  G4double TotalCrossSection(G4double t);

  // Controls
  inline void ExtendLowEnergyLimit(G4double threshold);
  inline void SetVerboseLevel(G4int verbose) { verboseLevel = verbose; }
  inline void SelectStationary(G4bool input) { statCode = input; }
  inline void UseAngularScattering(G4bool input) { fUseAngularScattering = input; }
  inline G4bool GetUseAngularScattering() const { return fUseAngularScattering; }

  // Vibrational mode info (centers and widths in eV)
  G4double VibrationEnergy(G4int level) const;     // center ω0(level) [eV] (for reference)
  G4double VibrationBWidth(G4int level) const;     // Gaussian b(level) [eV]

  // Sum of all partial cross sections at energy k
  inline G4double Sum(G4double k) const {
    G4double total = 0.0;
    for (G4int i = 0; i < (G4int)kNumLevels; ++i) {
      total += PartialCrossSection(k, i);
    }
    return total;
  }

  // Draw a vibrational energy loss ω from truncated Gaussian on [0, E0]
  G4double SampleVibrationalOmega(G4int level, G4double E0) const;

  // ---- Per-step diagnostics for logging (thread-local) ----
  // These expose, for the last sampled vibrational interaction on this thread:
  // - the chosen channel index (level 0..kNumLevels-1)
  // - the partial microscopic cross section at pre-step energy (cm^2)
  // - the pre-step kinetic energy (eV)
  static inline G4int    GetLastChannelIndex()      { return s_lastLevel; }
  static inline G4double GetLastPartialSigma_cm2()  { return s_lastSigmaPartial_cm2; }
  static inline G4double GetLastEkin_eV()           { return s_lastEkin_eV; }
  static inline void     ClearLastVibInfo()         { s_lastLevel = -1; s_lastSigmaPartial_cm2 = -1.0; s_lastEkin_eV = -1.0; }

 private:
  // ---- Random selection among levels (by partial σ) ----
  G4int RandomSelect(G4double energy);

  // ---- Angular scattering ----
  G4double RandomizeCosTheta(G4int level, G4double k);
  G4double Theta(G4int level, G4double k, G4double integrDiff);

  // ---- Utilities ----
  static inline G4double Clamp(G4double x, G4double lo, G4double hi)
  {
    return (x < lo ? lo : (x > hi ? hi : x));
  }

  // Fast initial guess for inv(erf), then Newton–Raphson refinement.
  // y must be in (-1,1). Returns x with erf(x) ≈ y.
  static G4double InvErfInitial(G4double y);
  static G4double InvErfRefine(G4double y, G4double x0);

  // Inverse CDF of a Gaussian with center mu and width b (no 1/2 in exponent),
  // truncated to [lo, hi]. u in (0,1).
  G4double TruncatedGaussianInvCDF(G4double u,
                                   G4double mu,
                                   G4double b,
                                   G4double lo,
                                   G4double hi) const;

  // Linear interpolation helper
  static G4double LinInterpolate(G4double e1, G4double e2, G4double e,
                                 G4double xs1, G4double xs2);
  
  static G4double LinLinInterpolate(G4double e1, G4double e2, G4double e,
                                    G4double xs1, G4double xs2);

  static G4double QuadInterpolator(G4double e11, G4double e12, G4double e21, G4double e22,
                                   G4double x11, G4double x12, G4double x21, G4double x22,
                                   G4double t1, G4double t2, G4double t, G4double e);

 private:
  // ----------------- Model configuration/state -----------------
  static constexpr G4int kNumLevels = 8;   // we use 8 vibrational modes (no 0.001 eV process)
  G4bool statCode = false;                 // keep electron direction & KE unchanged if true
  G4bool fUseAngularScattering = true;     // if true, use differential cross sections for angular scattering

  // Output change handler
  G4ParticleChangeForGamma* fParticleChangeForGamma = nullptr;

  // Water density table
  const std::vector<G4double>* fpWaterDensity = nullptr;

  // Init and verbosity
  G4bool isInitialised = false;
  G4int verboseLevel = 0;

  // Energy grid (eV) used in data file and corresponding per-level and total XS (in 1e-16 cm^2 units)
  std::vector<G4double> tdummyVec;                    // E-grid (eV) from the data file
  std::vector<std::vector<G4double>> fEnergyLevelXS;  // [iEnergy][level] partial σ in 1e-16 cm^2
  std::vector<G4double> fEnergyTotalXS;               // [iEnergy] total σ in 1e-16 cm^2

  // Gaussian broadening control
  G4bool fUseGaussianOmega = true;

  // Angular scattering data structures (one set per vibrational level)
  using VecMap = std::map<G4double, std::vector<G4double>>;
  using TriDimensionMap = std::map<G4double, std::map<G4double, G4double>>;
  std::array<VecMap, kNumLevels> eVecm;                      // [level][energy] -> vector of CDFs
  std::array<TriDimensionMap, kNumLevels> eDiffCrossSectionData;  // [level][energy][CDF] -> angle
  std::vector<G4double> eTdummyVec;                           // energy grid for angular data

  // Vibrational mode parameters from Michaud (centers and widths) in eV
  // (no invented spectra; widths are the Gaussian b = FWHM / (2*sqrt(ln 2)))
  std::array<G4double, kNumLevels> fOmega_eV = {
      0.024, 0.061, 0.092, 0.204, 0.417, 0.460, 0.510, 0.835
  };

  std::array<G4double, kNumLevels> fB_eV = {
      0.025 / 1.665,  // v_T'' (phonon)
      0.030 / 1.665,  // v_L' (lib1)
      0.040 / 1.665,  // v_L'' (lib2)
      0.016 / 1.665,  // v2 (bend)
      0.050 / 1.665,  // v1,3 (stretch)
      0.005 / 1.665,  // v3 (asym stretch)
      0.040 / 1.665,  // v1,3 + vL
      0.075 / 1.665   // 2*v1,3  (if you prefer 0.500 eV combo, adjust here)
  };

  // Thread-local snapshot for last sampled vib interaction, for logging
  static thread_local G4int    s_lastLevel;
  static thread_local G4double s_lastSigmaPartial_cm2;
  static thread_local G4double s_lastEkin_eV;
};

// ----------------- inline / small methods -----------------

inline void
G4DNAMichaudExcitationModel::ExtendLowEnergyLimit(G4double threshold)
{
  if (threshold < 2 * CLHEP::eV) {
    G4Exception("*** WARNING : the G4DNAMichaudExcitationModel class is not "
                "validated below 2 eV !", "", JustWarning, "");
  }
  SetLowEnergyLimit(threshold);
}

inline G4double
G4DNAMichaudExcitationModel::VibrationEnergy(G4int level) const
{
  // Center energy ω0 of the given vibrational level (eV)
  const G4int idx = (level < 0 ? 0 : (level >= kNumLevels ? kNumLevels - 1 : level));
  return fOmega_eV[idx] * CLHEP::eV;
}

inline G4double
G4DNAMichaudExcitationModel::VibrationBWidth(G4int level) const
{
  // Gaussian width parameter b (eV) of the given vibrational level
  const G4int idx = (level < 0 ? 0 : (level >= kNumLevels ? kNumLevels - 1 : level));
  return fB_eV[idx] * CLHEP::eV;
}

#endif // G4DNAMichaudExcitationModel_h
