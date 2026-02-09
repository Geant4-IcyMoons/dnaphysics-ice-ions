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
//

#include "G4DNAMichaud_ELSEPA_HIGH_ElasticModel.hh"
#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"
#include "G4DNAMolecularMaterial.hh"
#include "G4LogLogInterpolation.hh"
#include "G4Exp.hh"
#include "ModelDataRegistry.hh"

using namespace std;

#define MICHAUD_VERBOSE

G4DNAMichaud_ELSEPA_HIGH_ElasticModel::
G4DNAMichaud_ELSEPA_HIGH_ElasticModel(const G4ParticleDefinition*, const G4String& nam) :
    G4VEmModel(nam) 
{
  // Tables start at 200 eV and extend to 10 MeV
  SetLowEnergyLimit(200. * eV);
  SetHighEnergyLimit(10.0 * MeV);

  verboseLevel = 2;

#ifdef MICHAUD_VERBOSE
  if (verboseLevel > 0)
  {
    G4cout << "Michaud_ELSEPA_HIGH Elastic model is constructed "
           << G4endl
           << "Energy range: "
           << LowEnergyLimit() / eV << " eV - "
           << HighEnergyLimit() / MeV << " MeV"
           << G4endl;
  }
#endif
  
  fParticleChangeForGamma = nullptr;
  fpMolWaterDensity = nullptr;
  fpData = nullptr;
}

G4DNAMichaud_ELSEPA_HIGH_ElasticModel::~G4DNAMichaud_ELSEPA_HIGH_ElasticModel()
{
  delete fpData;
  eVecm.clear();
}

void G4DNAMichaud_ELSEPA_HIGH_ElasticModel::Initialise(const G4ParticleDefinition* particle,
                                           const G4DataVector& /*cuts*/)
{
#ifdef MICHAUD_VERBOSE
  if (verboseLevel > 3)
  {
    G4cout << "Calling G4DNAMichaud_ELSEPA_HIGH_ElasticModel::Initialise()" << G4endl;
  }
#endif
  
  if(particle->GetParticleName() != "e-")
  {
    G4Exception("G4DNAMichaud_ELSEPA_HIGH_ElasticModel::Initialise",
                "em0002",
                FatalException,
                "Model not applicable to particle type.");
  }

  if (LowEnergyLimit() < 200.*eV)
  {
    G4cout << "G4DNAMichaud_ELSEPA_HIGH_ElasticModel: low energy limit increased from "
           << LowEnergyLimit()/eV << " eV to " << 200. << " eV"
           << G4endl;
    SetLowEnergyLimit(200.*eV);
  }

  if (HighEnergyLimit() > 10.*MeV)
  {
    G4cout << "G4DNAMichaud_ELSEPA_HIGH_ElasticModel: high energy limit decreased from "
           << HighEnergyLimit()/eV << " eV to " << 1e7 << " eV"
           << G4endl;
    SetHighEnergyLimit(10.*MeV);
  }

  if (isInitialised) { return; }

  // Total cross section (10^-16 cm^2 units)
  G4double scaleFactor = 1e-16*cm*cm;
  G4String fileElectron("dna/sigma_elastic_e_michaud_elsepa_high");
  ModelDataRegistry::Instance().Record(
    std::string("model_ref:") + GetName(),
    ModelDataRegistry::NormalizeDatBasename(fileElectron));

  fpData = new G4DNACrossSectionDataSet(new G4LogLogInterpolation(),
                                        eV,
                                        scaleFactor );
  fpData->LoadData(fileElectron);

  const char *path = G4FindDataDir("G4LEDATA");

  if (path == nullptr)
  {
    G4Exception("G4DNAMichaud_ELSEPA_HIGH_ElasticModel::Initialise",
                "em0006",
                FatalException,
                "G4LEDATA environment variable not set.");
    return;
  }

  std::ostringstream eFullFileName;
  eFullFileName << path << "/dna/sigmadiff_cumulated_elastic_e_michaud_elsepa_high.dat";
  std::ifstream eDiffCrossSection(eFullFileName.str().c_str());

  if (!eDiffCrossSection)
  {
    G4ExceptionDescription errMsg;
    errMsg << "Missing data file:/dna/sigmadiff_cumulated_elastic_e_michaud_elsepa_high.dat";
    
    G4Exception("G4DNAMichaud_ELSEPA_HIGH_ElasticModel::Initialise",
                "em0003",
                FatalException,
                errMsg);
  }

  eTdummyVec.clear();
  eVecm.clear();
  eDiffCrossSectionData.clear();

  eTdummyVec.push_back(0.);

  while(!eDiffCrossSection.eof())
  {
    G4double tDummy;  // Energy
    G4double eDummy;  // Cumulative probability
    G4double angleDummy;  // Angle (degrees)
    
    eDiffCrossSection >> tDummy >> eDummy >> angleDummy;

    if (tDummy != eTdummyVec.back())
    {
      eTdummyVec.push_back(tDummy);
      eVecm[tDummy].push_back(0.);
    }

    eDiffCrossSectionData[tDummy][eDummy] = angleDummy;

    if (eDummy != eVecm[tDummy].back()) eVecm[tDummy].push_back(eDummy);
  }

#ifdef MICHAUD_VERBOSE
  if (verboseLevel>0)
  {
    if (verboseLevel > 1)
    {
      G4cout << "Loaded cross section files for Michaud_ELSEPA_HIGH Elastic model" << G4endl;
    }

    G4cout << "Michaud_ELSEPA_HIGH Elastic model is initialized " << G4endl
           << "Energy range: "
           << LowEnergyLimit() / eV << " eV - "
           << HighEnergyLimit() / MeV << " MeV"
           << G4endl;
  }
#endif

  G4DNAMolecularMaterial::Instance()->Initialize();
  fpMolWaterDensity = G4DNAMolecularMaterial::Instance()->
    GetNumMolPerVolTableFor(G4Material::GetMaterial("G4_WATER"));

  fParticleChangeForGamma = GetParticleChangeForGamma();
  isInitialised = true;
}

G4double
G4DNAMichaud_ELSEPA_HIGH_ElasticModel::
CrossSectionPerVolume(const G4Material* material,
#ifdef MICHAUD_VERBOSE
                      const G4ParticleDefinition* p,
#else
                      const G4ParticleDefinition*,
#endif
                      G4double ekin,
                      G4double,
                      G4double)
{
#ifdef MICHAUD_VERBOSE
  if (verboseLevel > 3)
  {
   G4cout << "Calling CrossSectionPerVolume() of G4DNAMichaud_ELSEPA_HIGH_ElasticModel"
          << G4endl;
  }
#endif

  G4double sigma = 0.;
  G4double waterDensity = (*fpMolWaterDensity)[material->GetIndex()];

  if (ekin <= HighEnergyLimit() && ekin >= LowEnergyLimit())
  {
      sigma = fpData->FindValue(ekin);
  }

#ifdef MICHAUD_VERBOSE
  if (verboseLevel > 2)
  {
    G4cout << "__________________________________" << G4endl;
    G4cout << "=== G4DNAMichaud_ELSEPA_HIGH_ElasticModel - XS INFO START" << G4endl;
    G4cout << "=== Kinetic energy(eV)=" << ekin/eV << " particle : " << p->GetParticleName() << G4endl;
    G4cout << "=== Cross section per water molecule (cm^2)=" << sigma/cm/cm << G4endl;
    G4cout << "=== Cross section per water molecule (cm^-1)=" << sigma*waterDensity/(1./cm) << G4endl;
    G4cout << "=== G4DNAMichaud_ELSEPA_HIGH_ElasticModel - XS INFO END" << G4endl;
  }
#endif

  return sigma*waterDensity;
}

void G4DNAMichaud_ELSEPA_HIGH_ElasticModel::SampleSecondaries(std::vector<G4DynamicParticle*>* /*fvect*/,
                                                  const G4MaterialCutsCouple* /*couple*/,
                                                  const G4DynamicParticle* aDynamicElectron,
                                                  G4double,
                                                  G4double)
{

#ifdef MICHAUD_VERBOSE
  if (verboseLevel > 3)
  {
    G4cout << "Calling SampleSecondaries() of G4DNAMichaud_ELSEPA_HIGH_ElasticModel" << G4endl;
  }
#endif

  G4double electronEnergy0 = aDynamicElectron->GetKineticEnergy();

  G4double cosTheta = RandomizeCosTheta(electronEnergy0);

  G4double phi = 2. * pi * G4UniformRand();

  G4ThreeVector zVers = aDynamicElectron->GetMomentumDirection();
  G4ThreeVector xVers = zVers.orthogonal();
  G4ThreeVector yVers = zVers.cross(xVers);

  G4double xDir = std::sqrt(1. - cosTheta*cosTheta);
  G4double yDir = xDir;
  xDir *= std::cos(phi);
  yDir *= std::sin(phi);

  G4ThreeVector zPrimeVers((xDir*xVers + yDir*yVers + cosTheta*zVers));

  fParticleChangeForGamma->ProposeMomentumDirection(zPrimeVers.unit());

  fParticleChangeForGamma->SetProposedKineticEnergy(electronEnergy0);

}

G4double G4DNAMichaud_ELSEPA_HIGH_ElasticModel::Theta(G4double k,
                                          G4double integrDiff)
{
  G4double theta = 0.;
  G4double valueT1 = 0;
  G4double valueT2 = 0;
  G4double valueE21 = 0;
  G4double valueE22 = 0;
  G4double valueE12 = 0;
  G4double valueE11 = 0;
  G4double xs11 = 0;
  G4double xs12 = 0;
  G4double xs21 = 0;
  G4double xs22 = 0;

  if (eTdummyVec.size() < 2) {
    G4Exception("G4DNAMichaud_ELSEPA_HIGH_ElasticModel::Theta",
                "em0003",
                FatalException,
                "Elastic data table is empty or invalid.");
  }
  auto t2 = std::upper_bound(eTdummyVec.begin(), eTdummyVec.end(), k);
  if (t2 == eTdummyVec.begin())
  {
    t2 = eTdummyVec.begin() + 1;
  }
  else if (t2 == eTdummyVec.end())
  {
    t2 = eTdummyVec.end() - 1; // clamp to last interval (e.g., k at upper bound)
  }
  auto t1 = t2 - 1;

  auto &vec1 = eVecm[(*t1)];
  if (vec1.size() < 2) {
    G4Exception("G4DNAMichaud_ELSEPA_HIGH_ElasticModel::Theta",
                "em0003",
                FatalException,
                "Elastic data table has insufficient CDF entries (t1).");
  }
  auto e12 = std::upper_bound(vec1.begin(), vec1.end(), integrDiff);
  if (e12 == vec1.end()) { e12 = vec1.end() - 1; }
  auto e11 = e12 - 1;

  auto &vec2 = eVecm[(*t2)];
  if (vec2.size() < 2) {
    G4Exception("G4DNAMichaud_ELSEPA_HIGH_ElasticModel::Theta",
                "em0003",
                FatalException,
                "Elastic data table has insufficient CDF entries (t2).");
  }
  auto e22 = std::upper_bound(vec2.begin(), vec2.end(), integrDiff);
  if (e22 == vec2.end()) { e22 = vec2.end() - 1; }
  auto e21 = e22 - 1;

  valueT1 = *t1;
  valueT2 = *t2;
  valueE21 = *e21;
  valueE22 = *e22;
  valueE12 = *e12;
  valueE11 = *e11;

  xs11 = eDiffCrossSectionData[valueT1][valueE11];
  xs12 = eDiffCrossSectionData[valueT1][valueE12];
  xs21 = eDiffCrossSectionData[valueT2][valueE21];
  xs22 = eDiffCrossSectionData[valueT2][valueE22];

  if (xs11 == 0 && xs12 == 0 && xs21 == 0 && xs22 == 0) return (0.);

  theta = QuadInterpolator(valueE11, valueE12, valueE21, valueE22, xs11, xs12,
                           xs21, xs22, valueT1, valueT2, k, integrDiff);

  return theta;
}

G4double G4DNAMichaud_ELSEPA_HIGH_ElasticModel::LinLogInterpolate(G4double e1,
                                                      G4double e2,
                                                      G4double e,
                                                      G4double xs1,
                                                      G4double xs2)
{
  G4double d1 = std::log(xs1);
  G4double d2 = std::log(xs2);
  G4double value = G4Exp(d1 + (d2 - d1) * (e - e1) / (e2 - e1));
  return value;
}

G4double G4DNAMichaud_ELSEPA_HIGH_ElasticModel::LinLinInterpolate(G4double e1,
                                                      G4double e2,
                                                      G4double e,
                                                      G4double xs1,
                                                      G4double xs2)
{
  G4double d1 = xs1;
  G4double d2 = xs2;
  G4double value = (d1 + (d2 - d1) * (e - e1) / (e2 - e1));
  return value;
}

G4double G4DNAMichaud_ELSEPA_HIGH_ElasticModel::LogLogInterpolate(G4double e1,
                                                      G4double e2,
                                                      G4double e,
                                                      G4double xs1,
                                                      G4double xs2)
{
  G4double a = (std::log10(xs2) - std::log10(xs1))
      / (std::log10(e2) - std::log10(e1));
  G4double b = std::log10(xs2) - a * std::log10(e2);
  G4double sigma = a * std::log10(e) + b;
  G4double value = (std::pow(10., sigma));
  return value;
}

G4double G4DNAMichaud_ELSEPA_HIGH_ElasticModel::QuadInterpolator(G4double e11,
                                                     G4double e12,
                                                     G4double e21,
                                                     G4double e22,
                                                     G4double xs11,
                                                     G4double xs12,
                                                     G4double xs21,
                                                     G4double xs22,
                                                     G4double t1,
                                                     G4double t2,
                                                     G4double t,
                                                     G4double e)
{
  G4double interpolatedvalue1 = LinLinInterpolate(e11, e12, e, xs11, xs12);
  G4double interpolatedvalue2 = LinLinInterpolate(e21, e22, e, xs21, xs22);
  G4double value = LinLinInterpolate(t1, t2, t, interpolatedvalue1,
                                     interpolatedvalue2);

  return value;
}

G4double G4DNAMichaud_ELSEPA_HIGH_ElasticModel::RandomizeCosTheta(G4double k)
{

  if (k < LowEnergyLimit() || k > HighEnergyLimit()) {
    G4Exception("G4DNAMichaud_ELSEPA_HIGH_ElasticModel::RandomizeCosTheta",
                "em0002",
                FatalException,
                "Elastic model called outside its energy limits.");
  }

  G4double integrdiff = 0;
  G4double uniformRand = G4UniformRand();
  integrdiff = uniformRand;

  G4double theta = 0.;
  G4double cosTheta = 0.;
  theta = Theta(k / eV, integrdiff);

  cosTheta = std::cos(theta * pi / 180);

  return cosTheta;
}

void G4DNAMichaud_ELSEPA_HIGH_ElasticModel::SetKillBelowThreshold(G4double)
{
  G4ExceptionDescription errMsg;
  errMsg << "The method G4DNAMichaud_ELSEPA_HIGH_ElasticModel::SetKillBelowThreshold is deprecated";
  
  G4Exception("G4DNAMichaud_ELSEPA_HIGH_ElasticModel::SetKillBelowThreshold",
              "deprecated",
              JustWarning,
              errMsg);
}
