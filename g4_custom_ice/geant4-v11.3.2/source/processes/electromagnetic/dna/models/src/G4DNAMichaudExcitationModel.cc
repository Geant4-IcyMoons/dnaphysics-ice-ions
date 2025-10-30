// //
// // ********************************************************************
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

// // Created by Z. Francis and adapted by G. Yoffe

// #include "G4DNAMichaudExcitationModel.hh"
// #include "G4SystemOfUnits.hh"
// #include "G4DNAMolecularMaterial.hh"

// using namespace std;

// G4DNAMichaudExcitationModel::G4DNAMichaudExcitationModel(const G4ParticleDefinition*, const G4String& nam) : G4VEmModel(nam)
// {
// 	fpWaterDensity = nullptr;
// 	SetLowEnergyLimit(2.*eV);
// 	SetHighEnergyLimit(100*eV);
// 	nLevels = 9;
// 	verboseLevel = 0;
// 	statCode = false;
// 	isInitialised = false;
// 	fParticleChangeForGamma = nullptr;
// }

// G4DNAMichaudExcitationModel::~G4DNAMichaudExcitationModel() = default;

// void G4DNAMichaudExcitationModel::Initialise(const G4ParticleDefinition* /*particle*/, const G4DataVector& /*cuts*/)
// {
// 	if (LowEnergyLimit() < 2.*eV)
// 	{
// 		G4Exception("*** WARNING : the G4DNAMichaudExcitationModel class is not "
// 								"validated below 2 eV !",
// 								"", JustWarning, "");
// 	}

// 	if (HighEnergyLimit() > 100.*eV)
// 	{
// 		G4cout << "G4DNAMichaudExcitationModel: high energy limit decreased from " <<
// 		HighEnergyLimit()/eV << " eV to " << 100. << " eV" << G4endl;
// 		SetHighEnergyLimit(100.*eV);
// 	}

// 	fpWaterDensity = G4DNAMolecularMaterial::Instance()->
// 			GetNumMolPerVolTableFor(G4Material::GetMaterial("G4_WATER"));

// 	if (isInitialised) {return;}

// 	fParticleChangeForGamma = GetParticleChangeForGamma();
// 	isInitialised = true;

// 	const char *path = G4FindDataDir("G4LEDATA");
// 	std::ostringstream eFullFileName;
// 	eFullFileName << path << "/dna/sigma_excitationvib_e_michaud.dat";
// 	std::ifstream input(eFullFileName.str().c_str());

// 	if (!input)
// 	{
// 		G4Exception("G4DNAMichaudExcitationModel::Initialise","em0003",
// 				FatalException,"Missing data file:/dna/sigma_excitationvib_e_michaud.dat");
// 	}

// 	tdummyVec.clear();

// 	G4double t;
// 	G4double xs;

// 	while(!input.eof())
// 	{
// 		input>>t;
// 		tdummyVec.push_back(t);

// 		fEnergyLevelXS.emplace_back();
// 		fEnergyTotalXS.push_back(0);
// 		std::vector<G4double>& levelXS = fEnergyLevelXS.back();
// 		levelXS.reserve(8);

// 		for(size_t i = 0 ; i < 8 ;++i)
// 		{
// 			input>>xs;
// 			levelXS.push_back(xs);
// 			fEnergyTotalXS.back() += xs;
// 		}
// 	}
// }

// G4double G4DNAMichaudExcitationModel::CrossSectionPerVolume(const G4Material* material,
// 																													 const G4ParticleDefinition*
// 																													 ,
// 																													 G4double ekin,
// 																													 G4double,
// 																													 G4double)
// {
// 	G4double sigma = 0.;
// 	G4double waterDensity = (*fpWaterDensity)[material->GetIndex()];
// 	if (ekin >= LowEnergyLimit() && ekin <= HighEnergyLimit())
// 		sigma =  TotalCrossSection(ekin);
// 	return sigma*2.*waterDensity;
// }

// void G4DNAMichaudExcitationModel::SampleSecondaries(std::vector< G4DynamicParticle*>*,
// 																									 const G4MaterialCutsCouple*,
// 																									 const G4DynamicParticle* aDynamicElectron,
// 																									 G4double,
// 																									 G4double)
// {
// 	G4double electronEnergy0 = aDynamicElectron->GetKineticEnergy();
// 	G4int level = RandomSelect(electronEnergy0);
// 	G4double excitationEnergy = VibrationEnergy(level);
// 	G4double newEnergy = electronEnergy0 - excitationEnergy;
// 	if (electronEnergy0 <= HighEnergyLimit() && newEnergy>0.)
// 	{
// 		if (!statCode)     
// 		{     
// 			fParticleChangeForGamma->ProposeMomentumDirection(aDynamicElectron->GetMomentumDirection());
// 			fParticleChangeForGamma->SetProposedKineticEnergy(newEnergy);
// 			fParticleChangeForGamma->ProposeLocalEnergyDeposit(excitationEnergy);
// 		}
// 		else 
// 		{
// 			fParticleChangeForGamma->ProposeMomentumDirection(aDynamicElectron->GetMomentumDirection());
// 			fParticleChangeForGamma->SetProposedKineticEnergy(electronEnergy0);
// 			fParticleChangeForGamma->ProposeLocalEnergyDeposit(excitationEnergy);
// 		}
// 	}
// }

// G4double G4DNAMichaudExcitationModel::PartialCrossSection(G4double t,
// 																												 G4int level)
// {
// 	if (t/eV==tdummyVec.back()) t=t*(1.-1e-12);
// 	auto t2 = std::upper_bound(tdummyVec.begin(),
// 																											tdummyVec.end(), t / eV);
// 	auto t1 = t2 - 1;
// 	size_t i1 = t1 - tdummyVec.begin();
// 	size_t i2 = t2 - tdummyVec.begin();
// 	G4double sigma = LinInterpolate((*t1), (*t2),
// 																t / eV,
// 																fEnergyLevelXS[i1][level],
// 																fEnergyLevelXS[i2][level]);
// 	// static const G4double conv_factor =  1e-16 * cm * cm;
//   static const G4double conv_factor =  1. * cm * cm;
// 	sigma *= conv_factor;
// 	if (sigma == 0.) sigma = 1e-30;
// 	return (sigma);
// }

// G4double G4DNAMichaudExcitationModel::TotalCrossSection(G4double t)
// {
// 	if (t/eV==tdummyVec.back()) t=t*(1.-1e-12);
// 	auto t2 = std::upper_bound(tdummyVec.begin(),
// 																											tdummyVec.end(), t / eV);
// 	auto t1 = t2 - 1;
// 	size_t i1 = t1 - tdummyVec.begin();
// 	size_t i2 = t2 - tdummyVec.begin();
// 	G4double sigma = LinInterpolate((*t1), (*t2),
// 																t / eV,
// 																fEnergyTotalXS[i1],
// 																fEnergyTotalXS[i2]);
// 	static const G4double conv_factor =  1e-16 * cm * cm;
// 	sigma *= conv_factor;
// 	if (sigma == 0.) sigma = 1e-30;
// 	return (sigma);
// }

// G4double G4DNAMichaudExcitationModel::VibrationEnergy(G4int level)
// {
// 	static G4double energies[8] = { 0.024, 0.061, 0.092, 0.204, 0.417, 0.460,
// 													 0.500, 0.835 };
// 	return (energies[level] * eV);
// }

// G4int G4DNAMichaudExcitationModel::RandomSelect(G4double k)
// {
// 	G4int i = nLevels;
// 	G4double value = 0.;
// 	std::deque<G4double> values;
// 	while (i > 0)
// 	{
// 		i--;
// 		G4double partial = PartialCrossSection(k, i);
// 		values.push_front(partial);
// 		value += partial;
// 	}
// 	value *= G4UniformRand();
// 	i = nLevels;
// 	while (i > 0)
// 	{
// 		i--;
// 		if (values[i] > value)
// 		{
// 			return i;
// 		}
// 		value -= values[i];
// 	}
// 	return 0;
// }

// G4double G4DNAMichaudExcitationModel::Sum(G4double k)
// {
// 	G4double totalCrossSection = 0.;
// 	for (G4int i = 0; i < nLevels; i++)
// 	{
// 		totalCrossSection += PartialCrossSection(k, i);
// 	}
// 	return totalCrossSection;
// }

// G4double G4DNAMichaudExcitationModel::LinInterpolate(G4double e1,
// 																										G4double e2,
// 																										G4double e,
// 																										G4double xs1,
// 																										G4double xs2)
// {
// 	G4double a = (xs2 - xs1) / (e2 - e1);
// 	G4double b = xs2 - a * e2;
// 	G4double value = a * e + b;
// 	return value;
// }
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under the terms  and *
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
// ********************************************************************
//
// Created by Z. Francis; adapted for Michaud vibrational Gaussian broadening by G. Yoffe
//

#include "G4DNAMichaudExcitationModel.hh"

#include "G4DynamicParticle.hh"
#include "G4Electron.hh"
#include "G4EMDataSet.hh"
#include "G4Material.hh"
#include "G4MaterialCutsCouple.hh"
#include "G4ParticleChangeForGamma.hh"
#include "G4RandomTools.hh"
#include "G4SystemOfUnits.hh"
#include "G4DNAMolecularMaterial.hh"

#include "Randomize.hh"          // G4UniformRand


#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <fstream>
#include <sstream>
#include <vector>

using std::size_t;

namespace {

// Inverse error function: good double-precision approximation with 2 Newton steps.
// Initial guess: Winitzki (a=0.147). Then refine.
inline G4double InvErf(G4double x)
{
  // clamp to domain
  if (x <= -1.) return -1e300;
  if (x >=  1.) return  1e300;
  // Winitzki approximation
  const G4double a  = 0.147;
  const G4double sgn= (x < 0.) ? -1.0 : 1.0;
  const G4double ln = std::log(1.0 - x*x);
  const G4double tt = 2.0/(G4double(CLHEP::pi)*a) + 0.5*ln;
  G4double y = sgn * std::sqrt( std::sqrt(tt*tt - ln/a) - tt );

  // 2x Newton-Raphson refinement on erf(y)-x = 0
  for (int k=0; k<2; ++k) {
    const G4double ey  = std::erf(y);
    const G4double dEy = 2.0/std::sqrt(CLHEP::pi) * std::exp(-y*y);
    const G4double dy  = (ey - x)/dEy;
    y -= dy;
  }
  return y;
}



} // anonymous namespace

// Static member function for linear interpolation
G4double G4DNAMichaudExcitationModel::LinInterpolate(G4double e1, G4double e2,
                                                     G4double e, G4double xs1, G4double xs2)
{
  const G4double a = (xs2 - xs1) / (e2 - e1);
  const G4double b = xs2 - a * e2;
  return a * e + b;
}

G4DNAMichaudExcitationModel::G4DNAMichaudExcitationModel(const G4ParticleDefinition*,
                                                         const G4String& nam)
: G4VEmModel(nam)
, fParticleChangeForGamma(nullptr)
, fpWaterDensity(nullptr)
{
  SetLowEnergyLimit(2.0*eV);
  SetHighEnergyLimit(100.0*eV);

  // Michaud uses 8 vibrational modes for this model (no "v_r" at ~1 meV here)
  // nLevels replaced by kNumLevels (header constant)

  verboseLevel = 0;
  statCode     = false;
  isInitialised= false;

  // Band centroids (eV) from Michaud
  fOmega_eV = { 0.024, 0.061, 0.092, 0.205, 0.417, 0.460, 0.510, 0.834 };

  // Gaussian width parameter b_i (eV) with the "no 1/2" convention:
  // g(ω) = 1/(sqrt(pi) b) * exp(-(ω-ω0)^2 / b^2)
  // These b are FWHM / 1.665 as used in your Python exporters.
  fB_eV = { 0.025/1.665, 0.030/1.665, 0.040/1.665, 0.016/1.665,
            0.050/1.665, 0.005/1.665, 0.040/1.665, 0.075/1.665 };

  fUseGaussianOmega = true; // broaden energy loss by default
}

G4DNAMichaudExcitationModel::~G4DNAMichaudExcitationModel() = default;


void G4DNAMichaudExcitationModel::Initialise(const G4ParticleDefinition*,
                                             const G4DataVector&)
{
  if (LowEnergyLimit() < 2.*eV) {
    G4Exception("*** WARNING : the G4DNAMichaudExcitationModel class is not validated below 2 eV !",
                "", JustWarning, "");
  }

  if (HighEnergyLimit() > 100.*eV) {
    G4cout << "G4DNAMichaudExcitationModel: high energy limit decreased from "
           << HighEnergyLimit()/eV << " eV to " << 100. << " eV" << G4endl;
    SetHighEnergyLimit(100.*eV);
  }

  fpWaterDensity = G4DNAMolecularMaterial::Instance()->
      GetNumMolPerVolTableFor(G4Material::GetMaterial("G4_WATER"));

  if (isInitialised) return;

  fParticleChangeForGamma = GetParticleChangeForGamma();
  isInitialised = true;

  // --- Load partial cross sections per vibrational level (units: 1e-16 cm^2) ---
  const char* path = G4FindDataDir("G4LEDATA");
  std::ostringstream eFullFileName;
  eFullFileName << path << "/dna/sigma_excitationvib_e_michaud.dat";

  std::ifstream input(eFullFileName.str().c_str());
  if (!input) {
    G4Exception("G4DNAMichaudExcitationModel::Initialise", "em0003",
                FatalException, "Missing data file:/dna/sigma_excitationvib_e_michaud.dat");
  }

  tdummyVec.clear();
  fEnergyLevelXS.clear();
  fEnergyTotalXS.clear();

  // Read: one energy, followed by 8 partial XS (in 1e-16 cm^2)
  while (true) {
    G4double E; // eV
    if (!(input >> E)) break;

    tdummyVec.push_back(E);
    fEnergyLevelXS.emplace_back();
    fEnergyTotalXS.push_back(0.0);

    std::vector<G4double>& levelXS = fEnergyLevelXS.back();
    levelXS.reserve(kNumLevels);

  for (size_t i = 0; i < kNumLevels; ++i) {
      G4double xs = 0.0;
      if (!(input >> xs)) { xs = 0.0; }
      levelXS.push_back(xs);        // still in table units (1e-16 cm^2)
      fEnergyTotalXS.back() += xs;  // sum in same units
    }
  }

  // Defensive: drop last row if it was partially read (e.g. trailing newline)
  if (!tdummyVec.empty() && fEnergyLevelXS.back().size() != kNumLevels) {
    tdummyVec.pop_back();
    fEnergyLevelXS.pop_back();
    fEnergyTotalXS.pop_back();
  }
}


G4double
G4DNAMichaudExcitationModel::CrossSectionPerVolume(const G4Material* material,
                                                   const G4ParticleDefinition*,
                                                   G4double ekin,
                                                   G4double, G4double)
{
  G4double sigma = 0.0;
  const G4double waterDensity = (*fpWaterDensity)[material->GetIndex()];
  if (ekin >= LowEnergyLimit() && ekin <= HighEnergyLimit()) {
    sigma = TotalCrossSection(ekin); // in cm^2
  }
  // factor 2.*waterDensity as in original model
  return sigma * 2.0 * waterDensity;
}


void
G4DNAMichaudExcitationModel::SampleSecondaries(std::vector<G4DynamicParticle*>*,
                                               const G4MaterialCutsCouple*,
                                               const G4DynamicParticle* aDynamicElectron,
                                               G4double, G4double)
{
  const G4double e0 = aDynamicElectron->GetKineticEnergy(); // eV
  const G4int    level = RandomSelect(e0);                  // 0..7

  // Sample energy loss for the chosen mode.
  G4double dE = 0.0;
  if (fUseGaussianOmega) {
    dE = SampleVibrationalOmega(level, e0);                 // in eV
  } else {
    dE = VibrationEnergy(level);                            // discrete (eV)
  }

  // Keep physical limits
  if (dE > e0) dE = 0.999999 * e0;
  if (dE < 0.0) dE = 0.0;

  const G4double newEnergy = e0 - dE;

  if (e0 <= HighEnergyLimit() && newEnergy > 0.0) {
    // Direction unchanged in this model
    fParticleChangeForGamma->ProposeMomentumDirection(aDynamicElectron->GetMomentumDirection());
    if (!statCode) {
      fParticleChangeForGamma->SetProposedKineticEnergy(newEnergy);
    } else {
      fParticleChangeForGamma->SetProposedKineticEnergy(e0);
    }
    fParticleChangeForGamma->ProposeLocalEnergyDeposit(dE);
  }
}


G4double
G4DNAMichaudExcitationModel::PartialCrossSection(G4double t, G4int level)
{
  // Interpolate partial cross section for a given mode
  // Input table units are 1e-16 cm^2; convert to cm^2.
  if (t/eV == tdummyVec.back()) t = t*(1.0 - 1e-12);

  auto it2 = std::upper_bound(tdummyVec.begin(), tdummyVec.end(), t / eV);
  auto it1 = it2 - 1;
  const size_t i1 = size_t(it1 - tdummyVec.begin());
  const size_t i2 = size_t(it2 - tdummyVec.begin());

  const G4double xs1 = fEnergyLevelXS[i1][level];
  const G4double xs2 = fEnergyLevelXS[i2][level];
  const G4double sig = LinInterpolate((*it1), (*it2), (t / eV), xs1, xs2);

  static const G4double conv = 1e-16 * cm * cm;
  G4double sigma = sig * conv;
  if (sigma <= 0.0) sigma = 1e-30;
  return sigma;
}


G4double
G4DNAMichaudExcitationModel::TotalCrossSection(G4double t)
{
  // Interpolate total cross section (sum of partial) from table (1e-16 cm^2)
  if (t/eV == tdummyVec.back()) t = t*(1.0 - 1e-12);

  auto it2 = std::upper_bound(tdummyVec.begin(), tdummyVec.end(), t / eV);
  auto it1 = it2 - 1;
  const size_t i1 = size_t(it1 - tdummyVec.begin());
  const size_t i2 = size_t(it2 - tdummyVec.begin());

  const G4double xs1 = fEnergyTotalXS[i1];
  const G4double xs2 = fEnergyTotalXS[i2];
  const G4double sig = LinInterpolate((*it1), (*it2), (t / eV), xs1, xs2);

  static const G4double conv = 1e-16 * cm * cm;
  G4double sigma = sig * conv;
  if (sigma <= 0.0) sigma = 1e-30;
  return sigma;
}





G4int
G4DNAMichaudExcitationModel::RandomSelect(G4double k)
{
  // Weighted selection by partial cross sections at kinetic energy k
  G4int i = (G4int)kNumLevels;
  G4double value = 0.0;
  std::deque<G4double> values;
  values.clear();
  values.resize(kNumLevels, 0.0);

  while (i > 0) {
    --i;
    const G4double partial = PartialCrossSection(k, i);
    values[i] = partial;
    value += partial;
  }

  value *= G4UniformRand();
  i = (G4int)kNumLevels;
  while (i > 0) {
    --i;
    if (values[i] > value) return i;
    value -= values[i];
  }
  return 0;
}





// ---- Gaussian sampling around each vibrational band (Michaud) ----
//
// g_i(ω) = 1/(sqrt(pi) b_i) * exp(-((ω-ω_i)^2)/b_i^2)   with ω, b_i in eV
// We sample ω from the truncated distribution on [0, E0/eV] by inverting the
// truncated CDF using an inverse-erf approximation.

G4double
G4DNAMichaudExcitationModel::SampleVibrationalOmega(G4int level, G4double electronEnergy0) const
{
  const G4double w0 = fOmega_eV[level]; // eV
  const G4double b  = fB_eV[level];     // eV
  const G4double E0 = electronEnergy0 / eV;

  // Normalization A over [0, E0] for the "no-half" Gaussian
  const G4double erfE0 = std::erf((E0 - w0) / b);
  const G4double erf0  = std::erf((-w0) / b); // = -erf(w0/b)
  const G4double A     = 0.5 * (erfE0 - erf0); // should be positive if E0 > 0

  if (!(A > 1e-16) || !(b > 0.0)) {
    // degenerate: fall back to centroid in [0, E0]
    G4double w = w0;
    if (w < 0.0) w = 0.0;
    if (w > E0 ) w = E0;
    return w * eV;
  }

  const G4double u   = G4UniformRand();           // (0,1)
  const G4double rhs = erf0 + 2.0 * u * A;        // target erf arg in (-1,1)
  // numeric safety
  const G4double rhsClamped = std::max(-0.999999999999, std::min(0.999999999999, rhs));
  const G4double inv        = InvErf(rhsClamped); // ≈ (ω - w0)/b
  G4double w = w0 + b * inv;

  // Clamp to physical range [0, E0]
  if (w < 0.0) w = 0.0;
  if (w > E0 ) w = E0;

  return w * eV;
}
