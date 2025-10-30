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
// ---------------------------------------------------------------------
//  G4EmDNAChemistry_ice.cc
//  Customized by G. Yoffe for ice (solid H2O) chemistry
//
//  This file defines a custom Geant4-DNA chemistry module for ice, using
//  ice-specific dissociation channels, Arrhenius constants, and chemical
//  processes. Differs from the original by modeling radiolysis and other
//  chemistry for ice, not liquid water.
// ---------------------------------------------------------------------
#include "G4EmDNAChemistry_ice.hh"
#include "G4ChemDissociationChannels.hh"
#include "G4ChemDissociationChannels_ice.hh"
#include "ArrheniusConstants_ice.hh"

#include "G4SystemOfUnits.hh"
#include <iomanip>

#include "G4DNAWaterDissociationDisplacer.hh"
#include "G4DNAChemistryManager.hh"
#include "G4ProcessManager.hh"

#include "G4DNAGenericIonsManager.hh"

// Forward declarations for Arrhenius debug functions
namespace G4ChemDissociationChannels_ice_debug {
  void SetIceChemistryInstance(G4EmDNAChemistry_ice* instance);
  G4double GetOHDiffusionAtPosition(G4double z_position);
}

// *** Processes and models for Geant4-DNA

#include "G4DNAVibExcitation.hh"
#include "G4DNASancheExcitationModel.hh"

#include "G4DNAMolecularDissociation.hh"
#include "G4DNAMolecularDissociation_ice.hh"
#include "G4DNABrownianTransportation_ice.hh"
#include "G4DNAMolecularReactionTable.hh"
#include "G4DNAMolecularStepByStepModel.hh"
#include "G4VDNAReactionModel.hh"
#include "G4DNASmoluchowskiReactionModel.hh"

#include "G4DNAElectronHoleRecombination.hh"
// particles

#include "G4Electron.hh"
#include "G4MoleculeTable.hh"
#include "G4H2O.hh"
#include "G4PhysicsListHelper.hh"

/****/
#include "G4DNAMoleculeEncounterStepper.hh"
#include "G4ProcessTable.hh"
#include "G4MolecularConfiguration.hh"
/****/

// factory
#include "G4PhysicsConstructorFactory.hh"

G4_DECLARE_PHYSCONSTR_FACTORY(G4EmDNAChemistry_ice);

#include "G4Threading.hh"

G4EmDNAChemistry_ice::G4EmDNAChemistry_ice() :
    G4VUserChemistryList(true)
{
  G4cout << "*** Using custom ice chemistry module! ***" << G4endl;
  G4DNAChemistryManager::Instance()->SetChemistryList(this);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void G4EmDNAChemistry_ice::SetTemperatureProfile(const std::vector<G4double>& temperatures)
{
  fTemperatures = temperatures;
  G4cout << "*** Ice chemistry temperature profile set: ";
  for(size_t i = 0; i < temperatures.size(); i++) {
    G4cout << temperatures[i] << " K";
    if(i < temperatures.size()-1) G4cout << ", ";
  }
  G4cout << " ***" << G4endl;
  
  // Set up Arrhenius debugging if depth profile is also available
  if (!fDepths.empty()) {
    DebugArrheniusSetup();
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void G4EmDNAChemistry_ice::SetDepthProfile(const std::vector<G4double>& depths)
{
  fDepths = depths;
  G4cout << "*** Ice chemistry depth profile set: ";
  for(size_t i = 0; i < depths.size(); i++) {
    G4cout << depths[i] << " mm";
    if(i < depths.size()-1) G4cout << ", ";
  }
  G4cout << " ***" << G4endl;
  
  // Set up Arrhenius debugging if temperature profile is also available
  if (!fTemperatures.empty()) {
    DebugArrheniusSetup();
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void G4EmDNAChemistry_ice::SetIceAmorphous(G4bool amorphous)
{
  fIceAmorphous = amorphous;
  G4cout << "*** Ice chemistry phase set to: " << (amorphous ? "amorphous" : "crystalline") << " ***" << G4endl;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4double G4EmDNAChemistry_ice::GetTemperatureAtPosition(G4double z_position) const
{
  // Temperature profile must be provided via macro - no defaults
  if (fTemperatures.empty() || fDepths.empty()) {
    G4Exception("G4EmDNAChemistry_ice::GetTemperatureAtPosition",
                "ICE_CHEM_001", FatalException,
                "No temperature profile set! Use /IcyMoons/detector/setZTemp in macro.");
  }
  
  // Find which depth layer this z-position corresponds to
  // For depths [0.0, 0.01, 0.02] and temps [100, 120]:
  // Layer 0: 0.0 <= z < 0.01 mm -> temperature[0] = 100K
  // Layer 1: 0.01 <= z < 0.02 mm -> temperature[1] = 120K
  // The fDepths array contains boundary points, fTemperatures contains layer values
  
  // Start from the second depth (layer boundary) and work backwards
  for (size_t i = 1; i < fDepths.size(); i++) {
    if (z_position < fDepths[i] * mm) {
      // z is in layer (i-1), so use temperature[i-1]
      size_t tempIndex = std::min(i - 1, fTemperatures.size() - 1);
      return fTemperatures[tempIndex] * kelvin;
    }
  }
  
  // If beyond all specified depths, use the last temperature
  return fTemperatures.back() * kelvin;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void G4EmDNAChemistry_ice::ConstructMolecule()
{
  // Use ice-specific diffusion coefficients instead of liquid water defaults
  G4ChemDissociationChannels_ice::ConstructMolecule();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void G4EmDNAChemistry_ice::ConstructDissociationChannels()
{
  // Use ice-specific dissociation channels
  G4ChemDissociationChannels_ice::ConstructDissociationChannels();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void G4EmDNAChemistry_ice::ConstructReactionTable(G4DNAMolecularReactionTable*
                                              theReactionTable)
{
  //-----------------------------------
  // Temperature profile is available via GetTemperatureAtPosition(z)
  // This can be used for future ice-specific reactions with Arrhenius temperature dependence
  //
  // Ice phase is available via GetIceAmorphous()
  // This can be used to implement different reaction mechanisms for amorphous vs crystalline ice
  //-----------------------------------
  
  //Get the molecular configuration
  G4MolecularConfiguration* OH =
   G4MoleculeTable::Instance()->GetConfiguration("°OH");
  G4MolecularConfiguration* OHm =
   G4MoleculeTable::Instance()->GetConfiguration("OHm");
  G4MolecularConfiguration* H2 =
   G4MoleculeTable::Instance()->GetConfiguration("H2");
  G4MolecularConfiguration* H3Op =
   G4MoleculeTable::Instance()->GetConfiguration("H3Op");
  G4MolecularConfiguration* H =
   G4MoleculeTable::Instance()->GetConfiguration("H");
  G4MolecularConfiguration* H2O2 =
   G4MoleculeTable::Instance()->GetConfiguration("H2O2");

  //------------------------------------------------------------------
  // *OH + *OH -> H2O2
  G4DNAMolecularReactionData* reactionData = new G4DNAMolecularReactionData(
      0.44e10 * (1e-3 * m3 / (mole * s)), OH, OH);
  reactionData->AddProduct(H2O2);
  theReactionTable->SetReaction(reactionData);
  //------------------------------------------------------------------
  // *OH + *H -> H2O
  theReactionTable->SetReaction(1.44e10 * (1e-3 * m3 / (mole * s)), OH, H);
  //------------------------------------------------------------------
  // *H + *H -> H2
  reactionData = new G4DNAMolecularReactionData(
      1.20e10 * (1e-3 * m3 / (mole * s)), H, H);
  reactionData->AddProduct(H2);
  theReactionTable->SetReaction(reactionData);
  //------------------------------------------------------------------
  // H3O+ + OH- -> 2H2O
  theReactionTable->SetReaction(1.43e11 * (1e-3 * m3 / (mole * s)), H3Op, OHm);
  //------------------------------------------------------------------
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void G4EmDNAChemistry_ice::ConstructProcess()
{
  auto pPhysicsListHelper = G4PhysicsListHelper::GetPhysicsListHelper();

  //===============================================================
  // Extend vibrational to low energy
  // Anyway, solvation of electrons is taken into account from 7.4 eV
  // So below this threshold, for now, no accurate modeling is done
  //
  G4VProcess* pProcess = G4ProcessTable::GetProcessTable()->
                                   FindProcess("e-_G4DNAVibExcitation", "e-");

  if (pProcess != nullptr)
  {
    G4DNAVibExcitation* pVibExcitation = (G4DNAVibExcitation*) pProcess;
    G4VEmModel* pModel = pVibExcitation->EmModel();
    G4DNASancheExcitationModel* pSancheExcitationMod =
        dynamic_cast<G4DNASancheExcitationModel*>(pModel);
    if(pSancheExcitationMod != nullptr)
    {
      pSancheExcitationMod->ExtendLowEnergyLimit(0.025 * eV);
    }
  }

  //===============================================================
  // Remove Electron Solvation process if it exists
  // This is necessary because G4EmDNAPhysics registers it by default
  // but we don't want hydrated electrons in our ice chemistry
  //
  pProcess = G4ProcessTable::GetProcessTable()->FindProcess("e-_G4DNAElectronSolvation", "e-");
  
  if (pProcess != nullptr)
  {
    G4cout << "*** Removing electron solvation process for ice chemistry ***" << G4endl;
    G4Electron::Definition()->GetProcessManager()->RemoveProcess(pProcess);
  }

  //===============================================================
  // Define processes for molecules
  //
  G4MoleculeTable* pMoleculeTable = G4MoleculeTable::Instance();
  G4MoleculeDefinitionIterator iterator = pMoleculeTable->GetDefintionIterator();
  iterator.reset();
  while (iterator())
  {
    G4MoleculeDefinition* pMoleculeDef = iterator.value();

    if (pMoleculeDef != G4H2O::Definition())
    {
      G4DNABrownianTransportation_ice* pBrownianTransport = new G4DNABrownianTransportation_ice();
      pPhysicsListHelper->RegisterProcess(pBrownianTransport, pMoleculeDef);
    }
    else
    {
      pMoleculeDef->GetProcessManager()->AddRestProcess(new G4DNAElectronHoleRecombination(), 2);
      G4DNAMolecularDissociation_ice* pDissociationProcess = new G4DNAMolecularDissociation_ice("H2O_DNAMolecularDecay_ice");
      pDissociationProcess->SetDisplacer(pMoleculeDef, new G4DNAWaterDissociationDisplacer);
      pDissociationProcess->SetVerboseLevel(2);  // Increased verbose level to see dissociation details

      G4cout << "H_CREATION_DEBUG: Ice-specific molecular dissociation process configured for " 
             << pMoleculeDef->GetParticleName() << " with verbose level 2" << G4endl;

      pMoleculeDef->GetProcessManager()->AddRestProcess(pDissociationProcess, 1);
    }
  }

  G4DNAChemistryManager::Instance()->Initialize();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void G4EmDNAChemistry_ice::DebugArrheniusSetup()
{
  G4cout << "\n";
  G4cout << "=================================================================" << G4endl;
  G4cout << "=                   ICE CHEMISTRY INITIALIZATION                =" << G4endl;
  G4cout << "=================================================================" << G4endl;
  G4cout << "=  Arrhenius Parameters:                                        =" << G4endl;
  G4cout << "=    OH: D₀ = " << std::setw(12) << std::scientific << std::setprecision(2) 
         << ArrheniusConstants_ice::D0_OH/(m2/s) << " m²/s, Eₐ = " << std::fixed << std::setprecision(3)
         << ArrheniusConstants_ice::Ea_OH/eV << " eV        =" << G4endl;
  G4cout << "=    H:  D₀ = " << std::setw(12) << std::scientific << std::setprecision(2)
         << ArrheniusConstants_ice::D0_H/(m2/s) << " m²/s, Eₐ = " << std::fixed << std::setprecision(3)
         << ArrheniusConstants_ice::Ea_H/eV << " eV         =" << G4endl;
  G4cout << "=================================================================" << G4endl;
  
  // Set the ice chemistry instance for position-dependent diffusion access
  G4ChemDissociationChannels_ice_debug::SetIceChemistryInstance(this);
  
  // Display detector layers with temperatures and calculated diffusion coefficients
  if (!fDepths.empty() && !fTemperatures.empty()) {
    G4cout << "=  Detector Layer Configuration:                               =" << G4endl;
    G4cout << "=================================================================" << G4endl;
    G4cout << "= Layer |    Depth Range    | Temp |    OH Diffusion    |    H Diffusion     =" << G4endl;
    G4cout << "=   #   |      (mm)         | (K)  |      (m²/s)        |      (m²/s)        =" << G4endl;
    G4cout << "=================================================================" << G4endl;
    
    for (size_t i = 0; i < fTemperatures.size(); ++i) {
      G4double tempK = fTemperatures[i];
      G4double startDepth = fDepths[i];           // Layer i starts at fDepths[i]
      G4double endDepth = fDepths[i+1];          // Layer i ends at fDepths[i+1]
      
      // Calculate diffusion coefficients at this temperature
      G4double D_OH = ArrheniusConstants_ice::CalculateArrheniusDiffusion(
          ArrheniusConstants_ice::D0_OH, ArrheniusConstants_ice::Ea_OH, tempK * kelvin);
      G4double D_H = ArrheniusConstants_ice::CalculateArrheniusDiffusion(
          ArrheniusConstants_ice::D0_H, ArrheniusConstants_ice::Ea_H, tempK * kelvin);
      
      G4cout << "=  " << std::setw(2) << (i+1) << "   | " 
             << std::fixed << std::setprecision(3) << std::setw(6) << startDepth/mm 
             << " - " << std::setw(6) << endDepth/mm << " | "
             << std::setw(4) << (int)tempK << " | "
             << std::scientific << std::setprecision(2) << std::setw(10) << D_OH/(m2/s) << " | "
             << std::setw(10) << D_H/(m2/s) << " =" << G4endl;
    }
    
    G4cout << "=================================================================" << G4endl;
  } else {
    G4cout << "=  WARNING: No temperature profile set! Using default values   =" << G4endl;
    G4cout << "=================================================================" << G4endl;
  }
  
  G4cout << "=  Status: Ice chemistry system initialized and ready          =" << G4endl;
  G4cout << "=================================================================" << G4endl;
  G4cout << "\n" << G4endl;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void G4EmDNAChemistry_ice::ConstructTimeStepModel(G4DNAMolecularReactionTable*
                                              reactionTable)
{
  G4VDNAReactionModel* reactionRadiusComputer = new G4DNASmoluchowskiReactionModel();
  reactionTable->PrintTable(reactionRadiusComputer);

  G4DNAMolecularStepByStepModel* stepByStep = new G4DNAMolecularStepByStepModel();
  stepByStep->SetReactionModel(reactionRadiusComputer);
//  ((G4DNAMoleculeEncounterStepper*) stepByStep->GetTimeStepper())->
//  SetVerbose(5);

  RegisterTimeStepModel(stepByStep, 0);
}
