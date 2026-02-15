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
// This example is provided by the Geant4-DNA collaboration
// Any report or published results obtained using the Geant4-DNA software
// shall cite the following Geant4-DNA collaboration publications:
// Med. Phys. 45 (2018) e722-e739
// Phys. Med. 31 (2015) 861-874
// Med. Phys. 37 (2010) 4692-4708
// Int. J. Model. Simul. Sci. Comput. 1 (2010) 157–178
//
// The Geant4-DNA web site is available at http://geant4-dna.org
//
/// \file PhysicsList.cc
/// \brief Implementation of the PhysicsList class

#include "PhysicsList.hh"

/*
#include "G4DecayPhysics.hh"
#include "G4EmDNAPhysics.hh"
#include "G4EmDNAPhysics_option1.hh"
#include "G4EmDNAPhysics_option2.hh"
#include "G4EmDNAPhysics_option3.hh"
#include "G4EmDNAPhysics_option4.hh"
#include "G4EmDNAPhysics_option5.hh"
#include "G4EmDNAPhysics_option6.hh"
#include "G4EmDNAPhysics_option7.hh"
#include "G4EmDNAPhysics_option8.hh"
#include "G4EmLivermorePhysics.hh"
#include "G4EmPenelopePhysics.hh"
#include "G4EmStandardPhysics.hh"
#include "G4EmStandardPhysics_option3.hh"
#include "G4EmStandardPhysics_option4.hh"
#include "G4RadioactiveDecayPhysics.hh"
*/

#include "G4EmBuilder.hh"
#include "G4EmDNAPhysics_option2.hh" // This is kept to construct all G4DNA particles
#include "G4EmParameters.hh"
#include "G4SystemOfUnits.hh"
#include "G4UserSpecialCuts.hh"
#include "G4PhysicsListHelper.hh"
#include "G4VEmProcess.hh"
#include "G4VEmModel.hh"
#include "G4ProcessManager.hh"

//****** ICE *****
#include "G4DNAVibExcitation.hh"
#include "G4DNAElastic.hh"
#include "G4DNAAttachment.hh"
#include "G4DNAExcitation.hh"
#include "G4DNAIonisation.hh"
#include "G4Electron.hh"
#include "G4DNASancheExcitationModel.hh"
#include "G4DNAMichaudExcitationModel.hh"
#include "G4DNAMichaudAttachmentModel.hh"
#include "G4DNAMichaudElasticModel.hh"
#include "G4DNAMichaud_ELSEPA_LOW_ElasticModel.hh"
#include "G4DNAMichaud_ELSEPA_HIGH_ElasticModel.hh"
#include "G4DNAEmfietzoglou_iceExcitationModel.hh"
#include "G4DNAEmfietzoglou_iceIonisationModel.hh"
#include "G4DNAElectronTrappingKill.hh"
//****** END ICE *****

//****** High-energy standard EM models (>10 MeV) *****
#include "G4eMultipleScattering.hh"
#include "G4eBremsstrahlung.hh"
#include "G4eIonisation.hh"
#include "G4MollerBhabhaModel.hh"
#include "G4SeltzerBergerModel.hh"
#include "G4UrbanMscModel.hh"
#include "G4WentzelVIModel.hh"
#include "G4CoulombScattering.hh"
//****** END High-energy *****

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

PhysicsList::PhysicsList() : G4VModularPhysicsList()
{
  SetDefaultCutValue(1.0 * micrometer);
  SetVerboseLevel(1);

  //fEmPhysics = "DNA_Opt2";

  fEmPhysicsList = new G4EmDNAPhysics_option2();
  
  //fDecayPhysicsList = new G4DecayPhysics();

  // Allow low-energy DNA processes (e.g., vib excitation ~2–100 eV)
  G4ProductionCutsTable::GetProductionCutsTable()->SetEnergyRange(1 * eV, 1 * GeV);
  G4EmParameters* param = G4EmParameters::Instance();
  param->SetMinEnergy(1 * eV);
  param->SetMaxEnergy(1 * GeV);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

PhysicsList::~PhysicsList()
{
  //delete fEmPhysicsList;
  //delete fDecayPhysicsList;
  //delete fRadDecayPhysicsList;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void PhysicsList::ConstructParticle()
{
  fEmPhysicsList->ConstructParticle();
  //fDecayPhysicsList->ConstructParticle();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void PhysicsList::ConstructProcess()
{
  AddTransportation();
  
  //****** ICE *****

  // Vibrational excitation process & model

G4PhysicsListHelper* ph = G4PhysicsListHelper::GetPhysicsListHelper();

// ----- Elastic (choose one block) -----
auto* theDNAElasticProcess = new G4DNAElastic("e-_G4DNAElastic_ICE");

// 1) Michaud only (1.7–100 eV)
// theDNAElasticProcess->SetEmModel(new G4DNAMichaudElasticModel());

// 2) Blended Michaud–ELSEPA low only (2–199 eV)
// auto* low = new G4DNAMichaud_ELSEPA_LOW_ElasticModel();
// low->SetLowEnergyLimit(2.*eV);
// low->SetHighEnergyLimit(199.*eV);
// theDNAElasticProcess->SetEmModel(low);

// 3) Blended Michaud–ELSEPA high only (200 eV–1 MeV)
// auto* high = new G4DNAMichaud_ELSEPA_HIGH_ElasticModel();
// high->SetLowEnergyLimit(200.*eV);
// high->SetHighEnergyLimit(10.*MeV);
// theDNAElasticProcess->SetEmModel(high);

// 4) Blended low + high tiled
theDNAElasticProcess->SetMinKinEnergy(2.*eV);
theDNAElasticProcess->SetMaxKinEnergy(10.*MeV);
auto* low = new G4DNAMichaud_ELSEPA_LOW_ElasticModel();
low->SetLowEnergyLimit(2.*eV);
low->SetHighEnergyLimit(200.*eV);
theDNAElasticProcess->SetEmModel(low); // sets default so SR is not re-inserted
auto* high = new G4DNAMichaud_ELSEPA_HIGH_ElasticModel();
high->SetLowEnergyLimit(200.*eV);
high->SetHighEnergyLimit(10.*MeV);
theDNAElasticProcess->AddEmModel(2, high);

ph->RegisterProcess(theDNAElasticProcess, G4Electron::ElectronDefinition());

// ----- Vibrational excitation (uncomment to enable) -----
auto* theDNAVibProcess = new G4DNAVibExcitation("e-_G4DNAVib_ICE");
theDNAVibProcess->SetEmModel(new G4DNAMichaudExcitationModel());
ph->RegisterProcess(theDNAVibProcess, G4Electron::ElectronDefinition());

// ----- Attachment (uncomment to enable) -----
auto* theDNAAttachmentProcess = new G4DNAAttachment("e-_G4DNAAttachment_ICE");
theDNAAttachmentProcess->SetEmModel(new G4DNAMichaudAttachmentModel());
ph->RegisterProcess(theDNAAttachmentProcess, G4Electron::ElectronDefinition());

// ----- Excitation -----
auto* theDNAExcitationProcess = new G4DNAExcitation("e-_G4DNAExcitation_ICE");
theDNAExcitationProcess->SetEmModel(new G4DNAEmfietzoglou_iceExcitationModel());
ph->RegisterProcess(theDNAExcitationProcess, G4Electron::ElectronDefinition());

  // ----- Ionisation -----
auto* theDNAIonisationProcess = new G4DNAIonisation("e-_G4DNAIonisation_ICE");
theDNAIonisationProcess->SetEmModel(new G4DNAEmfietzoglou_iceIonisationModel());
// Ensure ionisation is active across the DNA + high-EM tiled range.
// Without explicit bounds, process defaults may clip applicability.
theDNAIonisationProcess->SetMinKinEnergy(10. * eV);
theDNAIonisationProcess->SetMaxKinEnergy(1. * GeV);
ph->RegisterProcess(theDNAIonisationProcess, G4Electron::ElectronDefinition());

// ----- Electron trapping kill (<= 2 eV in Ice) -----
auto* theDNAElectronTrappingKill = new G4DNAElectronTrappingKill("e-_G4DNAElectronTrappingKill_ICE");
theDNAElectronTrappingKill->SetKillEnergyThreshold(2.0 * eV);
if (auto* ePM = G4Electron::ElectronDefinition()->GetProcessManager()) {
  ePM->AddDiscreteProcess(theDNAElectronTrappingKill);
}
  //****** END ICE *****

  // -------- High-energy fallback (>= 10 MeV): standard EM option4 models --------
  {
    const G4double kHighMin = 10. * MeV;
    const G4double kHighMax = 1. * GeV;

    // Multiple scattering (>= 10 MeV)
    // Note: MSC is a continuous process without SetMinKinEnergy; energy range controlled by model
    auto* msc = new G4eMultipleScattering();
    auto* mscModel = new G4UrbanMscModel();
    mscModel->SetLowEnergyLimit(kHighMin);
    mscModel->SetHighEnergyLimit(kHighMax);
    msc->SetEmModel(mscModel);
    ph->RegisterProcess(msc, G4Electron::ElectronDefinition());

    // Ionisation (>= 10 MeV) — must be a separate G4eIonisation
    // (G4VEnergyLossProcess) so that continuous dE/dx tables are built.
    // Tiling G4MollerBhabhaModel onto the G4DNAIonisation (G4VEmProcess)
    // would lose the continuous energy-loss component and prevent the
    // electron from slowing down through the 10 MeV boundary.
    // Both processes coexist: G4eIonisation is restricted to >= 10 MeV
    // by SetMinKinEnergy, and G4DNAIonisation covers 10 eV – 10 MeV.
    auto* highIonModel = new G4MollerBhabhaModel();
    highIonModel->SetLowEnergyLimit(kHighMin);
    highIonModel->SetHighEnergyLimit(kHighMax);
    auto* ionisation = new G4eIonisation();
    ionisation->SetMinKinEnergy(kHighMin);
    ionisation->SetEmModel(highIonModel);
    ph->RegisterProcess(ionisation, G4Electron::ElectronDefinition());

    // Bremsstrahlung (>= 10 MeV)
    auto* brem = new G4eBremsstrahlung();
    brem->SetMinKinEnergy(kHighMin);
    auto* bremModel = new G4SeltzerBergerModel();
    bremModel->SetLowEnergyLimit(kHighMin);
    bremModel->SetHighEnergyLimit(kHighMax);
    brem->SetEmModel(bremModel);
    ph->RegisterProcess(brem, G4Electron::ElectronDefinition());

    // Coulomb scattering (>= 10 MeV)
    auto* cs = new G4CoulombScattering();
    cs->SetMinKinEnergy(kHighMin);
    auto* csModel = new G4WentzelVIModel();
    csModel->SetLowEnergyLimit(kHighMin);
    csModel->SetHighEnergyLimit(kHighMax);
    cs->SetEmModel(csModel);
    ph->RegisterProcess(cs, G4Electron::ElectronDefinition());
  }

  /*
  fEmPhysicsList->ConstructProcess();
  fDecayPhysicsList->ConstructProcess();
  if (nullptr != fRadDecayPhysicsList) {
    fRadDecayPhysicsList->ConstructProcess();
  }
  if (fIsTrackingCutSet) {
    TrackingCut();
  }
  */
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void PhysicsList::AddPhysics(const G4String& name)
{
  /*
  if (name == fEmPhysics) {
    return;
  }

  G4cout << "### PhysicsList::AddPhysics Warning: Physics List <" << name << "> is requested"
         << G4endl;

  fEmPhysics = name;

  if (name == "emstandard_opt0") {
    delete fEmPhysicsList;
    fEmPhysicsList = new G4EmStandardPhysics();
  }
  else if (name == "emstandard_opt3") {
    delete fEmPhysicsList;
    fEmPhysicsList = new G4EmStandardPhysics_option3();
  }
  else if (name == "emstandard_opt4") {
    delete fEmPhysicsList;
    fEmPhysicsList = new G4EmStandardPhysics_option4();
  }
  else if (name == "raddecay") {
    if (nullptr == fRadDecayPhysicsList) fRadDecayPhysicsList = new G4RadioactiveDecayPhysics();
  }
  else if (name == "emlivermore") {
    delete fEmPhysicsList;
    fEmPhysicsList = new G4EmLivermorePhysics();
  }
  else if (name == "empenelope") {
    delete fEmPhysicsList;
    fEmPhysicsList = new G4EmPenelopePhysics();
  }
  else if (name == "DNA_Opt0") {
    delete fEmPhysicsList;
    fEmPhysicsList = new G4EmDNAPhysics();
  }
  else if (name == "DNA_Opt1") {
    delete fEmPhysicsList;
    fEmPhysicsList = new G4EmDNAPhysics_option1();
  }
  else if (name == "DNA_Opt2") {
    delete fEmPhysicsList;
    fEmPhysicsList = new G4EmDNAPhysics_option2();
  }
  else if (name == "DNA_Opt3") {
    delete fEmPhysicsList;
    fEmPhysicsList = new G4EmDNAPhysics_option3();
  }
  else if (name == "DNA_Opt4") {
    delete fEmPhysicsList;
    fEmPhysicsList = new G4EmDNAPhysics_option4();
  }
  else if (name == "DNA_Opt5") {
    delete fEmPhysicsList;
    fEmPhysicsList = new G4EmDNAPhysics_option5();
  }
  else if (name == "DNA_Opt6") {
    delete fEmPhysicsList;
    fEmPhysicsList = new G4EmDNAPhysics_option6();
  }
  else if (name == "DNA_Opt7") {
    delete fEmPhysicsList;
    fEmPhysicsList = new G4EmDNAPhysics_option7();
  }
  else if (name == "DNA_Opt8") {
    delete fEmPhysicsList;
    fEmPhysicsList = new G4EmDNAPhysics_option8();
  }
  else {
    G4cout << "### PhysicsList::AddPhysics Warning: Physics List <" << name
           << "> is does not exist - the command ignored" << G4endl;
  }
*/
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void PhysicsList::TrackingCut()
{
/*
  auto particle = G4GenericIon::GenericIon();  // DNA heavy ions
  auto particleName = particle->GetParticleName();
  auto capture = G4EmDNABuilder::FindOrBuildCapture(0.5 * CLHEP::MeV, particle);
  capture->AddRegion("World");
  capture->SetKinEnergyLimit(0.5 * CLHEP::MeV);  // 0.5 MeV/u
*/
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void PhysicsList::SetTrackingCut(G4bool isCut)
{
/*
  fIsTrackingCutSet = isCut;
*/
}
