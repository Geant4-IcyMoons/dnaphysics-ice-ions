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
// Based on the work described in
// Rad Res 163, 98-111 (2005)
// D. Emfietzoglou, H. Nikjoo
//
// Authors of the class (2014):
// I. Kyriakou (kyriak@cc.uoi.gr)
// D. Emfietzoglou (demfietz@cc.uoi.gr)
// S. Incerti (incerti@cenbg.in2p3.fr)
//

#include "G4DNAEmfietzoglouExcitationModelTracked.hh"
#include "G4SystemOfUnits.hh"
#include "G4DNAChemistryManager.hh"
#include "G4DNAMolecularMaterial.hh"
#include "ModelDataRegistry.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

using namespace std;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

namespace {
thread_local G4int g_lastExcitationLevelTracked = -1;
thread_local G4double g_lastExcitationSigma_cm2 = -1.0;
}

G4int G4DNAEmfietzoglouExcitationModelTracked::GetLastExcitationIndex()
{
  return g_lastExcitationLevelTracked;
}

void G4DNAEmfietzoglouExcitationModelTracked::ClearLastExcitationIndex()
{
  g_lastExcitationLevelTracked = -1;
  g_lastExcitationSigma_cm2 = -1.0;
}

G4double G4DNAEmfietzoglouExcitationModelTracked::GetLastPartialSigma_cm2()
{
  return g_lastExcitationSigma_cm2;
}

void G4DNAEmfietzoglouExcitationModelTracked::ClearLastPartialSigma_cm2()
{
  g_lastExcitationSigma_cm2 = -1.0;
}

G4DNAEmfietzoglouExcitationModelTracked::G4DNAEmfietzoglouExcitationModelTracked(const G4ParticleDefinition*,
                                                   const G4String& nam)
:G4VEmModel(nam)
{
    fpMolWaterDensity = nullptr;

    verboseLevel= 0;
    // Verbosity scale:
    // 0 = nothing
    // 1 = warning for energy non-conservation
    // 2 = details of energy budget
    // 3 = calculation of cross sections, file openings, sampling of atoms
    // 4 = entering in methods

    if( verboseLevel>0 )
    {
      G4cout << "Emfietzoglou excitation model is constructed " << G4endl;
    }
    fParticleChangeForGamma = nullptr;

    SetLowEnergyLimit(8.*eV);
    SetHighEnergyLimit(10.*keV);

    // Selection of stationary mode
    statCode = false;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

G4DNAEmfietzoglouExcitationModelTracked::~G4DNAEmfietzoglouExcitationModelTracked()
{
    // Cross section

    std::map< G4String,G4DNACrossSectionDataSet*,std::less<G4String> >::iterator pos;
    for (pos = tableData.begin(); pos != tableData.end(); ++pos)
    {
        G4DNACrossSectionDataSet* table = pos->second;
        delete table;
    }

}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void G4DNAEmfietzoglouExcitationModelTracked::Initialise(const G4ParticleDefinition* particle,
                                          const G4DataVector& /*cuts*/)
{

    if (verboseLevel > 3)
        G4cout << "Calling G4DNAEmfietzoglouExcitationModelTracked::Initialise()" << G4endl;

    G4String fileElectron("dna/sigma_excitation_e_emfietzoglou");
    ModelDataRegistry::Instance().Record(
      std::string("model_ref:") + GetName(),
      ModelDataRegistry::NormalizeDatBasename(fileElectron));

    G4ParticleDefinition* electronDef = G4Electron::ElectronDefinition();

    G4String electron;

    G4double scaleFactor = (1.e-22 / 3.343) * m*m;

    // *** ELECTRON

    electron = electronDef->GetParticleName();

    tableFile[electron] = fileElectron;

    // Cross section

    auto  tableE = new G4DNACrossSectionDataSet(new G4LogLogInterpolation, eV,scaleFactor );
    tableE->LoadData(fileElectron);

    tableData[electron] = tableE;

    //

    if( verboseLevel>0 )
    {
      G4cout << "Emfietzoglou excitation model is initialized " << G4endl
             << "Energy range: "
             << LowEnergyLimit() / eV << " eV - "
             << HighEnergyLimit() / keV << " keV for "
             << particle->GetParticleName()
             << G4endl;
    }

    // Initialize water density pointer
    fpMolWaterDensity = G4DNAMolecularMaterial::Instance()->GetNumMolPerVolTableFor(G4Material::GetMaterial("G4_WATER"));

    if (isInitialised) return;
    fParticleChangeForGamma = GetParticleChangeForGamma();
    isInitialised = true;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

G4double G4DNAEmfietzoglouExcitationModelTracked::CrossSectionPerVolume(const G4Material* material,
                                                         const G4ParticleDefinition* particleDefinition,
                                                         G4double ekin,
                                                         G4double,
                                                         G4double)
{
    if (verboseLevel > 3)
        G4cout << "Calling CrossSectionPerVolume() of G4DNAEmfietzoglouExcitationModelTracked" << G4endl;

    if (particleDefinition != G4Electron::ElectronDefinition()) return 0;

    // Calculate total cross section for model

    G4double sigma=0;

    G4double waterDensity = (*fpMolWaterDensity)[material->GetIndex()];

    const G4String& particleName = particleDefinition->GetParticleName();

    if (ekin >= LowEnergyLimit() && ekin <= HighEnergyLimit())
    {
      std::map< G4String,G4DNACrossSectionDataSet*,std::less<G4String> >::iterator pos;
      pos = tableData.find(particleName);

      if (pos != tableData.end())
      {
        G4DNACrossSectionDataSet* table = pos->second;
        if (table != nullptr) sigma = table->FindValue(ekin);
      }
      else
      {
        G4Exception("G4DNAEmfietzoglouExcitationModelTracked::CrossSectionPerVolume","em0002",
                            FatalException,"Model not applicable to particle type.");
      }
    }

    if (verboseLevel > 2)
    {
      G4cout << "__________________________________" << G4endl;
      G4cout << "G4DNAEmfietzoglouExcitationModelTracked - XS INFO START" << G4endl;
      G4cout << "Kinetic energy(eV)=" << ekin/eV << " particle : " << particleName << G4endl;
      G4cout << "Cross section per water molecule (cm^2)=" << sigma/cm/cm << G4endl;
      G4cout << "Cross section per water molecule (cm^-1)=" << sigma*waterDensity/(1./cm) << G4endl;
      //G4cout << "   Cross section per water molecule (cm^-1)=" <<
      ///sigma*material->GetAtomicNumDensityVector()[1]/(1./cm) << G4endl;
      G4cout << "G4DNAEmfietzoglouExcitationModelTracked - XS INFO END" << G4endl;
    }

    return sigma*waterDensity;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void G4DNAEmfietzoglouExcitationModelTracked::SampleSecondaries(std::vector<G4DynamicParticle*>* /*fvect*/,
                                                 const G4MaterialCutsCouple* /*couple*/,
                                                 const G4DynamicParticle* aDynamicParticle,
                                                 G4double,
                                                 G4double)
{

    if (verboseLevel > 3)
        G4cout << "Calling SampleSecondaries() of G4DNAEmfietzoglouExcitationModelTracked" << G4endl;

    G4double k = aDynamicParticle->GetKineticEnergy();

    const G4String& particleName = aDynamicParticle->GetDefinition()->GetParticleName();

    G4int level = RandomSelect(k,particleName);
    g_lastExcitationLevelTracked = level;
    G4double excitationEnergy = waterStructure.ExcitationEnergy(level);
    G4double newEnergy = k - excitationEnergy;

    if (newEnergy > 0)
    {
        fParticleChangeForGamma->ProposeMomentumDirection(aDynamicParticle->GetMomentumDirection());

        if (!statCode) fParticleChangeForGamma->SetProposedKineticEnergy(newEnergy);
        else fParticleChangeForGamma->SetProposedKineticEnergy(k);

        fParticleChangeForGamma->ProposeLocalEnergyDeposit(excitationEnergy);
    }

    const G4Track * theIncomingTrack = fParticleChangeForGamma->GetCurrentTrack();
    G4DNAChemistryManager::Instance()->CreateWaterMolecule(eExcitedMolecule,
                                                           level,
                                                           theIncomingTrack);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4int G4DNAEmfietzoglouExcitationModelTracked::RandomSelect(G4double k, const G4String& particle)
{

    G4int level = 0;
    g_lastExcitationSigma_cm2 = -1.0;

    std::map< G4String,G4DNACrossSectionDataSet*,std::less<G4String> >::iterator pos;
    pos = tableData.find(particle);

    if (pos != tableData.end())
    {
        G4DNACrossSectionDataSet* table = pos->second;

        if (table != nullptr)
        {
            auto  valuesBuffer = new G4double[table->NumberOfComponents()];
            const auto  n = (G4int)table->NumberOfComponents();
            G4int i(n);
            G4double value = 0.;

            //Check reading of initial xs file
	        //G4cout << table->GetComponent(0)->FindValue(k)/ ((1.e-22 / 3.343) * m*m) << G4endl;
            //G4cout << table->GetComponent(1)->FindValue(k)/ ((1.e-22 / 3.343) * m*m) << G4endl;
            //G4cout << table->GetComponent(2)->FindValue(k)/ ((1.e-22 / 3.343) * m*m) << G4endl;
            //G4cout << table->GetComponent(3)->FindValue(k)/ ((1.e-22 / 3.343) * m*m) << G4endl;
            //G4cout << table->GetComponent(4)->FindValue(k)/ ((1.e-22 / 3.343) * m*m) << G4endl;
            //G4cout << table->GetComponent(5)->FindValue(k)/ ((1.e-22 / 3.343) * m*m) << G4endl;
            //G4cout << table->GetComponent(6)->FindValue(k)/ ((1.e-22 / 3.343) * m*m) << G4endl;
            //abort();

	    while (i>0)
            {
                i--;
                valuesBuffer[i] = table->GetComponent(i)->FindValue(k);
                value += valuesBuffer[i];
            }

            value *= G4UniformRand();

            i = n;

            while (i > 0)
            {
                i--;

                if (valuesBuffer[i] > value)
                {
                    g_lastExcitationSigma_cm2 = valuesBuffer[i] / (cm * cm);
                    delete[] valuesBuffer;
                    return i;
                }
                value -= valuesBuffer[i];
            }

            delete[] valuesBuffer;

        }
    }
    else
    {
        G4Exception("G4DNAEmfietzoglouExcitationModelTracked::RandomSelect","em0002",
                    FatalException,"Model not applicable to particle type.");
    }
    return level;
}
